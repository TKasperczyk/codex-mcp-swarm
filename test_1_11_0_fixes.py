#!/usr/bin/env python3
"""
Regression tests for the 1.11.0 structured-failure fixes.

Covers terminal-event precedence, clean provider failure causes, model and
retryability reporting, non-terminal error events, warnings, and the sync and
reply paths. Drives the real handlers with subprocesses faked only for the two
sync cases. Every task artifact lives under a fresh temporary directory.
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
import time


TEST_ROOT = tempfile.mkdtemp(prefix="codex_swarm_test_111_")
atexit.register(shutil.rmtree, TEST_ROOT, True)
os.environ["CODEX_SWARM_TASK_DIR"] = os.path.join(TEST_ROOT, "tasks")
os.environ["CODEX_SWARM_WORKTREE_DIR"] = os.path.join(TEST_ROOT, "worktrees")
os.environ["CODEX_SWARM_LOG"] = os.path.join(TEST_ROOT, "server.log")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codex_mcp_swarm as m  # noqa: E402


PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


class Captured:
    """Swap out _send and collect every frame the server tries to write."""

    def __enter__(self):
        self.frames = []
        self._real = m._send
        m._send = self.frames.append
        return self

    def __exit__(self, *args):
        m._send = self._real

    def responses(self):
        return [frame for frame in self.frames if "result" in frame or "error" in frame]


def text_of(response):
    return "\n".join(
        content.get("text", "")
        for content in response.get("result", {}).get("content", [])
    )


_request_id = 100


def call(tool, arguments):
    global _request_id
    _request_id += 1
    with Captured() as captured:
        m._handle({
            "jsonrpc": "2.0",
            "id": _request_id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        })
    return text_of(captured.responses()[0])


def event(event_type, **values):
    return json.dumps({"type": event_type, **values})


THREAD_ID = "019c1234-5678-7000-8000-000000000111"
THREAD = event("thread.started", thread_id=THREAD_ID)
TURN_STARTED = event("turn.started")
TURN_COMPLETED = event("turn.completed", usage={})


def failed_event(message):
    return event("turn.failed", error={"message": message})


def agent_event(text):
    return event(
        "item.completed",
        item={"id": "item_1", "type": "agent_message", "text": text},
    )


def error_event(message):
    return event("error", message=message)


def warning_event(message):
    return event(
        "item.completed",
        item={"id": "item_warning", "type": "error", "message": message},
    )


m.TASK_DIR.mkdir(parents=True, exist_ok=True)
expected_task_dir = os.path.join(TEST_ROOT, "tasks")
if os.path.realpath(m.TASK_DIR) != os.path.realpath(expected_task_dir):
    raise RuntimeError(f"refusing to test against non-temporary task dir: {m.TASK_DIR}")

MADE = []
DEAD_PID, BAD_START = os.getpid(), 1.0
DEFAULT_COMMAND = "codex exec --json -c model=gpt-5.6-sol prompt"


def make_task(task_id, stdout, stderr="", exit_code=None, command=DEFAULT_COMMAND):
    meta = {
        "task_id": task_id,
        "status": "running",
        "started_at": time.time() - 5,
        "pid": DEAD_PID,
        "pid_start_time": BAD_START,
        "command": command,
    }
    (m.TASK_DIR / f"{task_id}.meta").write_text(json.dumps(meta))
    (m.TASK_DIR / f"{task_id}.stdout").write_text(stdout)
    (m.TASK_DIR / f"{task_id}.stderr").write_text(stderr)
    if exit_code is not None:
        # Simulate the real SIGCHLD path preserving an observed exit code.
        m._ASYNC_PIDS[DEAD_PID] = exit_code
    else:
        # No handler entry plus a non-child PID forces the exit-code-lost path.
        m._ASYNC_PIDS.pop(DEAD_PID, None)
    MADE.append(task_id)


class FakePopen:
    """Completed Popen used to exercise synchronous handler paths."""

    next_pid = 990000

    def __init__(self, command, stdout_text, stderr_text, exit_code):
        type(self).next_pid += 1
        self.pid = type(self).next_pid
        self.command = command
        self.stdout_text = stdout_text
        self.stderr_text = stderr_text
        self.returncode = exit_code

    def communicate(self, timeout=None):
        return self.stdout_text, self.stderr_text


def fake_popen_factory(stdout, stderr, exit_code, calls):
    def factory(command, **kwargs):
        calls.append(list(command))
        return FakePopen(command, stdout, stderr, exit_code)

    return factory


CAPACITY = "Selected model is at capacity. Please try a different model."
REVOKED = "Your refresh token has been revoked. Please sign out and sign in again."
RUST_NOISE = "\n".join(
    f"2026-08-31T12:00:{index:02d}Z ERROR codex_api::stream: escaped=\\\"noise-{index}\\\""
    for index in range(60)
) + "\n"


try:
    # -- 1. turn.failed beats noisy stderr and supplies the primary cause ----
    print("\n[1] turn.failed with noisy stderr")
    make_task(
        "11000001",
        "\n".join([THREAD, TURN_STARTED, failed_event(CAPACITY)]) + "\n",
        stderr=RUST_NOISE,
        exit_code=1,
    )
    output = call("codex_wait", {"task_ids": ["11000001"], "timeout": 1})
    check("structured capacity message is the cause", f"Cause: {CAPACITY}" in output)
    check("model is parsed from -c model=", "Model: gpt-5.6-sol" in output)
    check("capacity is classified retryable", "Category: provider_capacity" in output and "Retryable: yes" in output)
    check("stderr is explicitly secondary", "Diagnostics (stderr tail; not the failure cause):" in output)
    check("cause precedes stderr diagnostics", output.index("Cause:") < output.index("Diagnostics (stderr"))
    check("raw failure JSONL is not returned", '"type": "turn.failed"' not in output)

    # -- 2. empty stderr still reports the turn.failed cause ----------------
    print("\n[2] turn.failed with empty stderr")
    make_task(
        "11000002",
        "\n".join([THREAD, TURN_STARTED, failed_event(REVOKED)]) + "\n",
        stderr="",
        exit_code=1,
    )
    output = call("codex_wait", {"task_ids": ["11000002"], "timeout": 1})
    check("revoked-token cause survives empty stderr", f"Cause: {REVOKED}" in output)
    check("authentication failure is not retryable", "Category: authentication" in output and "Retryable: no" in output)
    check("no raw JSONL dump is used as fallback", '"type": "thread.started"' not in output)
    status_output = call("codex_status", {"task_ids": ["11000002"]})
    check("status uses the same cause and model", f"Cause: {REVOKED}" in status_output and "Model: gpt-5.6-sol" in status_output)
    cancel_output = call("codex_cancel", {"task_id": "11000002"})
    check("cancel on an already-failed task preserves the cause", f"Cause: {REVOKED}" in cancel_output)
    task_resource = json.loads(m._read_resource("codex-swarm:///tasks"))
    resource_entry = next(item for item in task_resource if item["task_id"] == "11000002")
    check("task resource exposes structured failure fields", resource_entry["failure_cause"] == REVOKED and resource_entry["model"] == "gpt-5.6-sol")

    # -- 3. turn.failed resolves an exit-code-lost run without inference ----
    print("\n[3] exit code lost after turn.failed")
    make_task(
        "11000003",
        "\n".join([THREAD, TURN_STARTED, failed_event(CAPACITY)]) + "\n",
    )
    state = m._resolve_task_state("11000003")
    output = call("codex_wait", {"task_ids": ["11000003"], "timeout": 1})
    check("terminal failure fixes the old completed verdict", state["status"] == "failed", state["status"])
    check("lost exit code is recorded", state["exit_code_lost"] is True)
    check("terminal verdict is observed, not inferred", state["status_inferred"] is False)
    check("wait omits the inference warning", "inferred, not observed" not in output)

    # -- 4. turn.failed beats an earlier partial agent message --------------
    print("\n[4] partial agent message followed by turn.failed")
    make_task(
        "11000004",
        "\n".join([
            THREAD,
            TURN_STARTED,
            agent_event("Partial answer that must not become a success."),
            failed_event(REVOKED),
        ]) + "\n",
    )
    info = m._check_task("11000004")
    output = call("codex_wait", {"task_ids": ["11000004"], "timeout": 1})
    status_output = call("codex_status", {"task_ids": ["11000004"]})
    check("partial message cannot override turn.failed", info["status"] == "failed", info["status"])
    check("wait reports the terminal cause", f"Cause: {REVOKED}" in output)
    # The invariant is that partial text is never THE RESULT -- not that it is
    # unmentionable. It must not appear before the labeled line, which is what
    # would make a caller read it as the outcome.
    check(
        "partial text is only ever labeled, never the result",
        "Partial answer" not in output.split("Partial output (before failure):")[0],
    )
    check(
        "wait keeps the labeled partial output",
        "Partial output (before failure): Partial answer" in output,
    )
    check("status labels rather than promotes partial output", "Partial output (before failure):" in status_output)

    # -- 5. a retry diagnostic followed by turn.completed is success --------
    print("\n[5] top-level error followed by turn.completed")
    make_task(
        "11000005",
        "\n".join([
            THREAD,
            TURN_STARTED,
            error_event("stream disconnected before completion; reconnecting"),
            agent_event("Recovered and completed successfully."),
            TURN_COMPLETED,
        ]) + "\n",
        stderr=RUST_NOISE,
        exit_code=1,
    )
    state = m._resolve_task_state("11000005")
    output = call("codex_wait", {"task_ids": ["11000005"], "timeout": 1})
    check("turn.completed wins over earlier error and noisy stderr", state["status"] == "completed", state["status"])
    check("terminal completion also wins over a conflicting exit", state["exit_code"] == 1 and state["status_inferred"] is False)
    check("successful assistant result is returned", "Recovered and completed successfully." in output)
    check("non-terminal error does not leak as FAILED", "FAILED" not in output)

    error_only = m._analyze_jsonl(error_event("temporary provider error") + "\n")
    error_only_zero = m._decide_run_status(0, error_only, "")
    error_only_nonzero = m._decide_run_status(1, error_only, "")
    error_only_lost = m._decide_run_status(None, error_only, "")
    check("top-level error alone is never terminal", error_only["terminal_status"] is None)
    check("error-only stream defers to a zero exit", error_only_zero == ("completed", False), repr(error_only_zero))
    check("error-only stream defers to a nonzero exit", error_only_nonzero == ("failed", False), repr(error_only_nonzero))
    check("error-only stream cannot manufacture an inferred failure", error_only_lost == ("completed", True), repr(error_only_lost))

    # -- 6. turn.failed supplies cause after an earlier top-level error ------
    print("\n[6] top-level error followed by turn.failed")
    make_task(
        "11000006",
        "\n".join([
            THREAD,
            TURN_STARTED,
            error_event("temporary stream error that was retried"),
            failed_event(REVOKED),
        ]) + "\n",
        exit_code=0,
    )
    state = m._resolve_task_state("11000006")
    output = call("codex_wait", {"task_ids": ["11000006"], "timeout": 1})
    check("turn.failed wins over a conflicting zero exit", state["status"] == "failed", state["status"])
    check("terminal cause beats earlier diagnostic error", f"Cause: {REVOKED}" in output and "Cause: temporary stream" not in output)
    check("terminal authentication classification is retained", "Category: authentication" in output)

    contradictory = m._analyze_jsonl(
        "\n".join([TURN_COMPLETED, failed_event(REVOKED)]) + "\n"
    )
    zero_status = m._decide_run_status(0, contradictory, "")
    nonzero_status = m._decide_run_status(1, contradictory, "")
    lost_status = m._decide_run_status(None, contradictory, "")
    check("both terminal types are detected as contradictory", contradictory["terminal_conflict"] is True and contradictory["terminal_status"] is None)
    check("zero exit resolves contradictory terminals as completed", zero_status == ("completed", False), repr(zero_status))
    check("nonzero exit resolves contradictory terminals as failed", nonzero_status == ("failed", False), repr(nonzero_status))
    check("lost exit makes contradictory-terminal fallback inferred", lost_status[1] is True, repr(lost_status))

    # -- 7. item.completed/type=error is a warning, never terminal ----------
    print("\n[7] non-terminal item error warning")
    warning = "Requested model was unavailable; using the configured fallback."
    warning_stdout = "\n".join([
        THREAD,
        TURN_STARTED,
        warning_event(warning),
        agent_event("Warning handled; run completed."),
    ]) + "\n"
    make_task("11000007", warning_stdout, exit_code=0)
    analysis = m._analyze_jsonl(warning_stdout)
    state = m._resolve_task_state("11000007")
    status_output = call("codex_status", {"task_ids": ["11000007"]})
    check("error item is collected as a warning", analysis["warnings"] == [warning])
    check("error item is not a terminal or top-level error", not analysis["terminal_events"] and not analysis["errors"])
    check("observed zero exit completes the run", state["status"] == "completed", state["status"])
    check("status labels the warning without reporting failure", f"Warning: {warning}" in status_output and "FAILED" not in status_output)

    # No terminal and no exit code is necessarily a compatibility inference.
    # It must remain visibly different from the terminal verdicts above.
    inferred_id = "1100000a"
    make_task(
        inferred_id,
        "\n".join([THREAD, TURN_STARTED, agent_event("Legacy completion evidence.")]) + "\n",
    )
    inferred_state = m._resolve_task_state(inferred_id)
    inferred_output = call("codex_wait", {"task_ids": [inferred_id], "timeout": 1})
    check("fallback completion is marked inferred", inferred_state["status_inferred"] is True)
    check("caller sees the inferred-completion warning", "completion verdict is inferred, not observed" in inferred_output)

    # -- 8. synchronous codex uses the same structured failure formatter ----
    print("\n[8] synchronous codex exit 1")
    sync_stdout = "\n".join([THREAD, TURN_STARTED, failed_event(CAPACITY)]) + "\n"
    calls = []
    real_popen = m.subprocess.Popen
    m.subprocess.Popen = fake_popen_factory(sync_stdout, RUST_NOISE, 1, calls)
    try:
        output = call("codex", {"prompt": "test", "model": "gpt-5.6-sol"})
    finally:
        m.subprocess.Popen = real_popen
    check("sync failure reports clean structured cause", f"Cause: {CAPACITY}" in output)
    check("sync failure reports actual model", "Model: gpt-5.6-sol" in output)
    check("sync stderr is secondary diagnostics", "Diagnostics (stderr tail; not the failure cause):" in output)
    check("sync wrapper launches no retry", len(calls) == 1, f"{len(calls)} launches")
    check("sync command requests public JSONL", bool(calls) and "--json" in calls[0])

    # -- 9. synchronous reply requests JSONL and reports the same details ----
    print("\n[9] synchronous codex_reply exit 1")
    reply_stdout = "\n".join([THREAD, TURN_STARTED, failed_event(REVOKED)]) + "\n"
    calls = []
    saved_config = dict(m.SERVER_CONFIG)
    real_popen = m.subprocess.Popen
    m.SERVER_CONFIG.clear()
    m.SERVER_CONFIG["model"] = "gpt-5.6-sol"
    m.subprocess.Popen = fake_popen_factory(reply_stdout, RUST_NOISE, 1, calls)
    try:
        output = call("codex_reply", {"threadId": THREAD_ID, "prompt": "continue"})
    finally:
        m.subprocess.Popen = real_popen
        m.SERVER_CONFIG.clear()
        m.SERVER_CONFIG.update(saved_config)
    check("reply failure reports clean structured cause", f"Cause: {REVOKED}" in output)
    check("reply failure reports configured model", "Model: gpt-5.6-sol" in output)
    check("reply failure says authentication is not retryable", "Category: authentication" in output and "Retryable: no" in output)
    check("reply wrapper launches no retry", len(calls) == 1, f"{len(calls)} launches")
    expected_prefix = ["codex", "exec", "resume", "--json"]
    check("reply requests JSONL in the accepted position", bool(calls) and calls[0][:4] == expected_prefix, repr(calls[0][:4] if calls else []))

finally:
    m._ASYNC_PIDS.pop(DEAD_PID, None)
    m._ASYNC_PROCS.pop(DEAD_PID, None)
    for task_id in MADE:
        for suffix in (".meta", ".stdout", ".stderr"):
            (m.TASK_DIR / f"{task_id}{suffix}").unlink(missing_ok=True)



# ---------------------------------------------------------------- 1.11.0 (b)
# codex_status labeled partial output from the start; codex_wait dropped it.
# The wait path is where results are actually collected, so that is exactly
# where "the agent had already started editing" has to be visible.
print("\npartial output reaches the collection path")

PARTIAL_TEXT = "I have started editing foo.py"
PARTIAL_TASK = "bbbb0001"
make_task(
    PARTIAL_TASK,
    stdout="\n".join([
        THREAD,
        event("turn.started"),
        agent_event(PARTIAL_TEXT),
        failed_event("Selected model is at capacity. Please try a different model."),
    ]),
    exit_code=1,
)

partial_wait = call("codex_wait", {"task_ids": [PARTIAL_TASK]})
check(
    "codex_wait surfaces partial output before a failure",
    f"Partial output (before failure): {PARTIAL_TEXT}" in partial_wait,
)
check(
    "codex_wait still leads with the structured cause",
    "Cause:" in partial_wait
    and partial_wait.index("Cause:") < partial_wait.index("Partial output"),
)

partial_info = m._check_task(PARTIAL_TASK)
check(
    "failed tasks expose partial_output",
    partial_info.get("partial_output") == PARTIAL_TEXT,
)
check("dead duplicate stderr key is gone", "stderr" not in partial_info)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for failure in FAIL:
        print(f"  FAILED: {failure}")
sys.exit(1 if FAIL else 0)

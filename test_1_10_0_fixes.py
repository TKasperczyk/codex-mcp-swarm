#!/usr/bin/env python3
"""
Regression tests for the 1.10.0 fixes.

Covers the spurious-failure verdict: when the exit code was lost to a reaping
race, the server used to mark a task "failed" purely because stderr was
non-empty -- which codex writes to on perfectly healthy runs. Finished work
was reported as failed and callers acted on it. Also covers the hints that
now ride along with unverified failures and with waits that time out.

Drives the real _handle()/_resolve_task_state with _send captured, so no
Codex process is ever spawned.
"""
import os
import sys
import json
import time

os.environ.setdefault("CODEX_SWARM_TASK_DIR", "/tmp/codex_swarm_tasks_test_110")
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

    def __exit__(self, *a):
        m._send = self._real

    def responses(self):
        return [f for f in self.frames if "result" in f or "error" in f]


def text_of(resp):
    return "\n".join(
        c.get("text", "") for c in resp.get("result", {}).get("content", [])
    )


m.TASK_DIR.mkdir(parents=True, exist_ok=True)
MADE = []

# A live PID paired with a start time that cannot match forces _is_alive to
# report death (the PID-reuse guard), while os.waitpid on a non-child raises
# ChildProcessError -- exactly the race that loses an exit code.
DEAD_PID, BAD_START = os.getpid(), 1.0

AGENT_MSG = json.dumps({
    "type": "item.completed",
    "item": {"type": "agent_message", "text": "Refactor complete. 80 tests passing."},
})
THREAD_LINE = json.dumps({"type": "thread.started", "thread_id": "t-abc"})
NOISY_STDERR = "WARN: sandbox denied /etc/shadow\nprogress: 40%\n"


def make_task(tid, stdout="", stderr="", exit_code=None, status=None):
    meta = {"task_id": tid, "started_at": time.time() - 30,
            "pid": DEAD_PID, "pid_start_time": BAD_START}
    if status:
        meta["status"] = status
        meta["completed_at"] = time.time()
    if exit_code is not None:
        meta["exit_code"] = exit_code
    (m.TASK_DIR / f"{tid}.meta").write_text(json.dumps(meta))
    (m.TASK_DIR / f"{tid}.stdout").write_text(stdout)
    (m.TASK_DIR / f"{tid}.stderr").write_text(stderr)
    MADE.append(tid)


def call(tool, args, rid=1):
    with Captured() as cap:
        m._handle({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                   "params": {"name": tool, "arguments": args}})
        return text_of(cap.responses()[0])


try:
    # -- 1. THE FIX: finished work + noisy stderr must not read as failure ----
    print("\n[1] lost exit code, noisy stderr, but the run produced a result")
    make_task("aaaa0001", stdout=f"{THREAD_LINE}\n{AGENT_MSG}\n", stderr=NOISY_STDERR)
    st = m._resolve_task_state("aaaa0001")
    check("status is completed, not failed", st["status"] == "completed", st["status"])
    check("marked as an inferred verdict", st["exit_code_lost"] is True)
    check("the agent's result survives", "80 tests passing" in m._check_task("aaaa0001")["result"])

    # -- 2. fallback preserved when there is genuinely no output -------------
    print("\n[2] lost exit code, stderr only, no agent message")
    make_task("aaaa0002", stdout="", stderr="Traceback: boom\n")
    st = m._resolve_task_state("aaaa0002")
    check("still falls back to failed", st["status"] == "failed", st["status"])
    check("marked as an inferred verdict", st["exit_code_lost"] is True)

    # -- 3. an observed non-zero exit is untouched ---------------------------
    print("\n[3] genuine exit 1, previously finalized")
    make_task("aaaa0003", stdout="", stderr="real failure\n", exit_code=1, status="failed")
    st = m._resolve_task_state("aaaa0003")
    check("stays failed", st["status"] == "failed", st["status"])
    check("NOT flagged as inferred", st["exit_code_lost"] is False)

    # -- 4. the flag persists across re-reads --------------------------------
    print("\n[4] flag persists to metadata")
    persisted = json.loads((m.TASK_DIR / "aaaa0002.meta").read_text())
    check("exit_code_lost written to meta", persisted.get("exit_code_lost") is True)
    check("re-read still reports it", m._resolve_task_state("aaaa0002")["exit_code_lost"] is True)

    # -- 5. codex_wait output carries the hint only when warranted ----------
    print("\n[5] codex_wait output")
    inferred = call("codex_wait", {"task_ids": ["aaaa0002"], "timeout": 1})
    check("inferred failure carries the verify hint", "inferred, not observed" in inferred)
    check("hint names a concrete check", "pgrep -af 'codex exec'" in inferred)
    genuine = call("codex_wait", {"task_ids": ["aaaa0003"], "timeout": 1})
    check("genuine failure has no hint", "inferred, not observed" not in genuine)

    # -- 6. a wait that times out is not a failure --------------------------
    print("\n[6] codex_wait timeout wording")
    real_floor = m._WAIT_MIN_TIMEOUT
    m._WAIT_MIN_TIMEOUT = 1
    try:
        alive = "aaaa0004"
        (m.TASK_DIR / f"{alive}.meta").write_text(json.dumps(
            {"task_id": alive, "started_at": time.time(), "pid": os.getpid()}))
        (m.TASK_DIR / f"{alive}.stdout").write_text("")
        (m.TASK_DIR / f"{alive}.stderr").write_text("")
        MADE.append(alive)
        out = call("codex_wait", {"task_ids": [alive], "timeout": 1})
        check("still reports STILL RUNNING", "STILL RUNNING" in out)
        check("says plainly it is not a failure", "NOT a failure" in out)
    finally:
        m._WAIT_MIN_TIMEOUT = real_floor

    # -- 7. codex_status shows the same caveat ------------------------------
    print("\n[7] codex_status output")
    status_out = call("codex_status", {"task_ids": ["aaaa0002"]})
    check("status flags the inferred verdict", "inferred, not observed" in status_out)

    # -- 8. tool descriptions carry what the client cannot infer ------------
    print("\n[8] tool descriptions")
    desc = {t["name"]: t["description"] for t in m.TOOLS}
    check("codex warns about the client-side ceiling", "300s" in desc["codex"])
    check("codex points at the async path", "codex_async" in desc["codex"])
    check("codex_status says it collects nothing", "collects nothing" in desc["codex_status"])
    check("codex_status warns FAILED is not authoritative",
          "not always authoritative" in desc["codex_status"])

finally:
    for tid in MADE:
        for suffix in (".meta", ".stdout", ".stderr"):
            (m.TASK_DIR / f"{tid}{suffix}").unlink(missing_ok=True)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
sys.exit(1 if FAIL else 0)

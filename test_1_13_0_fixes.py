#!/usr/bin/env python3
"""
Comprehensive tests for the 1.13.0 MCP protocol upgrade and reliability fixes.

Runnable via either ``pytest test_1_13_0_fixes.py`` or ``python3 test_1_13_0_fixes.py``.
Uses isolated temporary directories and fake python subprocesses (no real Codex
API calls) to exercise real process groups, real stdin pipes, concurrent metadata
writes, incremental JSONL caching, MCP 2026-07-28 stateless discovery, and the
official ``io.modelcontextprotocol/tasks`` extension end-to-end.
"""
import ast
import atexit
import inspect
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

TEST_ROOT = tempfile.mkdtemp(prefix="codex_swarm_test_113_")
atexit.register(shutil.rmtree, TEST_ROOT, True)
os.environ["CODEX_SWARM_TASK_DIR"] = os.path.join(TEST_ROOT, "tasks")
os.environ["CODEX_SWARM_WORKTREE_DIR"] = os.path.join(TEST_ROOT, "worktrees")
os.environ["CODEX_SWARM_LOG"] = os.path.join(TEST_ROOT, "server.log")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codex_mcp_swarm as m  # noqa: E402


class Captured:
    """Swap out _send and collect every frame the server writes."""

    def __enter__(self):
        self.frames = []
        self._real = m._send
        lock = threading.Lock()

        def fake(resp):
            with lock:
                self.frames.append(resp)

        m._send = fake
        return self

    def __exit__(self, *args):
        m._send = self._real

    def responses(self):
        return [f for f in self.frames if "result" in f or "error" in f]

    def progress(self):
        return [f for f in self.frames if f.get("method") == "notifications/progress"]


_REQ_ID = 1000


def _next_id():
    global _REQ_ID
    _REQ_ID += 1
    return _REQ_ID


def send_req(method, params=None, rid=None):
    if rid is None:
        rid = _next_id()
    req = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        req["params"] = params
    with Captured() as cap:
        m._handle(req)
    return cap


def make_fake_codex_script(name, body):
    path = Path(TEST_ROOT) / name
    path.write_text(body, encoding="utf-8")
    return path


def with_fake_codex(script_path):
    """Context manager redirecting ``_exec_argv`` to invoke a Python script."""

    class _Ctx:
        def __enter__(self):
            self._real = m._exec_argv

            def fake_exec(cmd):
                if cmd and cmd[0] == "codex":
                    return [sys.executable, str(script_path)] + cmd[1:]
                return cmd

            m._exec_argv = fake_exec
            return self

        def __exit__(self, *args):
            m._exec_argv = self._real

    return _Ctx()


def validate_against_schema(value, schema, path="$"):
    """Strict recursive JSON Schema subset validator for outputSchema checks."""
    expected_type = schema.get("type")
    if expected_type == "object":
        assert isinstance(value, dict), f"{path}: expected object, got {type(value).__name__}"
        for req_key in schema.get("required", []):
            assert req_key in value, f"{path}: missing required key {req_key!r}"
        props = schema.get("properties", {})
        for k, v in value.items():
            assert k in props, f"{path}: unexpected key {k!r} not in outputSchema properties"
            validate_against_schema(v, props[k], f"{path}.{k}")
    elif expected_type == "array":
        assert isinstance(value, list), f"{path}: expected array, got {type(value).__name__}"
        item_schema = schema.get("items")
        if item_schema:
            for idx, elem in enumerate(value):
                validate_against_schema(elem, item_schema, f"{path}[{idx}]")
    elif expected_type == "string":
        assert isinstance(value, str), f"{path}: expected string, got {type(value).__name__}"
        if "enum" in schema:
            assert value in schema["enum"], f"{path}: {value!r} not in {schema['enum']}"
    elif expected_type == "integer":
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"{path}: expected integer, got {type(value).__name__}"
        )
    elif expected_type == "boolean":
        assert isinstance(value, bool), f"{path}: expected boolean, got {type(value).__name__}"


# ===================================================================
# 1. MCP 2026-07-28 Stateless Discovery & Per-Request Metadata
# ===================================================================

def test_server_discover_2026_07_28():
    cap = send_req(
        "server/discover",
        {
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientCapabilities": {},
            }
        },
    )
    resp = cap.responses()[0]
    assert "result" in resp
    res = resp["result"]
    assert res["resultType"] == "complete"
    assert "2026-07-28" in res["supportedVersions"]
    assert "2025-11-25" in res["supportedVersions"]
    assert "io.modelcontextprotocol/tasks" in res["capabilities"]["extensions"]
    assert res["serverInfo"]["name"] == "codex-mcp-swarm"
    assert res["serverInfo"]["version"] == m.__version__
    assert res["_meta"]["io.modelcontextprotocol/serverInfo"]["version"] == m.__version__
    assert isinstance(res["instructions"], str) and "HEAD" in res["instructions"]
    assert res["ttlMs"] > 0
    assert res["cacheScope"] == "public"


def test_initialize_backward_compat_and_instructions():
    cap = send_req("initialize", {"protocolVersion": "2025-11-25", "capabilities": {}})
    res = cap.responses()[0]["result"]
    assert res["protocolVersion"] == "2025-11-25"
    assert "resultType" not in res
    assert isinstance(res["instructions"], str) and len(res["instructions"]) > 50
    assert "HEAD" in res["instructions"]


def test_stateless_result_type_and_unsupported_version():
    meta_2026 = {"_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}}

    # 2026-07-28 requests get resultType: "complete"
    for method, extra in [
        ("ping", {}),
        ("tools/list", {}),
        ("resources/list", {}),
        ("resources/read", {"uri": "codex-swarm:///server-info"}),
    ]:
        res = send_req(method, {**extra, **meta_2026}).responses()[0]["result"]
        assert res.get("resultType") == "complete", f"{method} missing resultType"

    # Legacy requests omit resultType
    for method, extra in [
        ("ping", {}),
        ("tools/list", {}),
        ("resources/list", {}),
        ("resources/read", {"uri": "codex-swarm:///server-info"}),
    ]:
        res = send_req(method, extra).responses()[0]["result"]
        assert "resultType" not in res, f"legacy {method} should not include resultType"

    # Unsupported per-request version returns -32022 with supported versions & requested
    err_resp = send_req(
        "tools/list",
        {"_meta": {"io.modelcontextprotocol/protocolVersion": "2099-01-01"}},
    ).responses()[0]
    assert "error" in err_resp
    assert err_resp["error"]["code"] == -32022
    assert "2026-07-28" in err_resp["error"]["data"]["supportedVersions"]
    assert "2026-07-28" in err_resp["error"]["data"]["supported"]
    assert err_resp["error"]["data"]["requested"] == "2099-01-01"


def test_ping_and_silent_unknown_notifications():
    # Ping works
    res = send_req("ping").responses()[0]["result"]
    assert res == {}

    # Unknown notification WITHOUT id -> ZERO frames sent (JSON-RPC 2.0 §4.1)
    with Captured() as cap:
        m._handle({"jsonrpc": "2.0", "method": "notifications/roots/list_changed", "params": {}})
    assert cap.frames == [], f"expected no reply to notification, got {cap.frames}"

    # Unknown notification WITH id: None -> ZERO frames sent
    with Captured() as cap:
        m._handle({"jsonrpc": "2.0", "id": None, "method": "unknown/notification"})
    assert cap.frames == [], f"expected no reply to id=None notification, got {cap.frames}"

    # Unknown request WITH non-null id -> -32601 error
    err = send_req("unknown/method").responses()[0]["error"]
    assert err["code"] == -32601


# ===================================================================
# 2. Tool Annotations, outputSchema, structuredContent, isError
# ===================================================================

def test_tool_annotations_and_output_schemas():
    tools = send_req("tools/list").responses()[0]["result"]["tools"]
    by_name = {t["name"]: t for t in tools}
    assert set(by_name) == {
        "codex", "codex_async", "codex_reply", "codex_status", "codex_wait", "codex_cancel"
    }

    for name, tool in by_name.items():
        ann = tool.get("annotations")
        assert isinstance(ann, dict), f"{name} missing annotations"
        for hint in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"):
            assert isinstance(ann.get(hint), bool), f"{name} missing bool annotation {hint}"
        assert isinstance(tool.get("outputSchema"), dict), f"{name} missing outputSchema"
        assert tool["outputSchema"]["type"] == "object"

    # Verify read-only hints are truthful
    for ro in ("codex_status", "codex_wait"):
        assert by_name[ro]["annotations"]["readOnlyHint"] is True
        assert by_name[ro]["annotations"]["destructiveHint"] is False
        assert by_name[ro]["annotations"]["openWorldHint"] is False
    for mut in ("codex", "codex_async", "codex_reply"):
        assert by_name[mut]["annotations"]["readOnlyHint"] is False
        assert by_name[mut]["annotations"]["openWorldHint"] is True
    assert by_name["codex_cancel"]["annotations"]["idempotentHint"] is True
    assert by_name["codex_cancel"]["annotations"]["openWorldHint"] is False

    # Worktree caveat documented in schema, instructions, and README
    wt_desc = by_name["codex"]["inputSchema"]["properties"]["worktree"]["description"]
    assert "HEAD" in wt_desc and "uncommitted" in wt_desc.lower()
    readme_text = (Path(__file__).resolve().parent / "README.md").read_text(encoding="utf-8")
    assert "uncommitted" in readme_text.lower() and "HEAD" in readme_text

    # codex_reply accepts cwd, model, sandbox, approval-policy, config
    reply_props = by_name["codex_reply"]["inputSchema"]["properties"]
    for key in ("threadId", "prompt", "cwd", "model", "sandbox", "approval-policy", "config"):
        assert key in reply_props, f"codex_reply missing property {key}"


def test_is_error_and_structured_content_sync_and_tools():
    tools_by_name = {t["name"]: t for t in m.TOOLS}

    ok_script = make_fake_codex_script(
        "fake_ok.py",
        """
import json, sys
prompt = sys.stdin.read()
print(json.dumps({"type": "thread.started", "thread_id": "t-ok-123"}))
print(json.dumps({"type": "turn.started"}))
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": f"ECHO:{prompt}"}}))
print(json.dumps({"type": "turn.completed"}))
""",
    )
    fail_script = make_fake_codex_script(
        "fake_fail.py",
        """
import json, sys
_ = sys.stdin.read()
print(json.dumps({"type": "thread.started", "thread_id": "t-fail-456"}))
print(json.dumps({"type": "turn.started"}))
print(json.dumps({"type": "turn.failed", "error": {"message": "Selected model is at capacity."}}))
sys.exit(1)
""",
    )

    # Sync codex success -> isError False + structuredContent matches outputSchema
    with with_fake_codex(ok_script):
        res = send_req(
            "tools/call",
            {"name": "codex", "arguments": {"prompt": "hello world", "model": "gpt-6-astra"}},
        ).responses()[0]["result"]
    assert res["isError"] is False
    assert "ECHO:hello world" in res["content"][0]["text"]
    validate_against_schema(res["structuredContent"], tools_by_name["codex"]["outputSchema"])
    assert res["structuredContent"]["status"] == "completed"
    assert res["structuredContent"]["thread_id"] == "t-ok-123"
    assert res["structuredContent"]["model"] == "gpt-6-astra"

    # Sync codex failure -> isError True + structuredContent failure object
    with with_fake_codex(fail_script):
        res = send_req(
            "tools/call",
            {"name": "codex", "arguments": {"prompt": "hello", "model": "gpt-6-astra"}},
        ).responses()[0]["result"]
    assert res["isError"] is True
    validate_against_schema(res["structuredContent"], tools_by_name["codex"]["outputSchema"])
    assert res["structuredContent"]["status"] == "failed"
    assert res["structuredContent"]["failure"]["category"] == "provider_capacity"
    assert res["structuredContent"]["failure"]["retryable"] is True

    # Sync codex_reply failure -> isError True + structuredContent
    with with_fake_codex(fail_script):
        res = send_req(
            "tools/call",
            {
                "name": "codex_reply",
                "arguments": {"threadId": "t-fail-456", "prompt": "try again", "model": "gpt-6-astra"},
            },
        ).responses()[0]["result"]
    assert res["isError"] is True
    validate_against_schema(res["structuredContent"], tools_by_name["codex_reply"]["outputSchema"])
    assert res["structuredContent"]["status"] == "failed"
    assert res["structuredContent"]["model"] == "gpt-6-astra"

    # codex_cancel on nonexistent task -> isError True
    res = send_req(
        "tools/call", {"name": "codex_cancel", "arguments": {"task_id": "deadbeef"}}
    ).responses()[0]["result"]
    assert res["isError"] is True
    validate_against_schema(res["structuredContent"], tools_by_name["codex_cancel"]["outputSchema"])
    assert res["structuredContent"]["status"] == "not_found"

    # codex_status on nonexistent task -> isError True
    res = send_req(
        "tools/call", {"name": "codex_status", "arguments": {"task_ids": ["deadbeef"]}}
    ).responses()[0]["result"]
    assert res["isError"] is True
    validate_against_schema(res["structuredContent"], tools_by_name["codex_status"]["outputSchema"])


# ===================================================================
# 3. Official MCP Tasks Extension (io.modelcontextprotocol/tasks)
# ===================================================================

def test_mcp_tasks_extension_lifecycle_complete_fail_and_cancel():
    tasks_meta = {
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": {
                "extensions": {"io.modelcontextprotocol/tasks": {}},
            },
        }
    }

    # --- A. Running -> Completed lifecycle with live statusMessage ---
    gate_file = Path(TEST_ROOT) / "task_gate"
    stream_script = make_fake_codex_script(
        "fake_stream.py",
        f"""
import json, sys, time
from pathlib import Path
prompt = sys.stdin.read()
print(json.dumps({{"type": "thread.started", "thread_id": "t-mcp-001"}}), flush=True)
print(json.dumps({{"type": "turn.started"}}), flush=True)
print(json.dumps({{"type": "item.started", "item": {{"type": "command_execution", "command": "pytest -q"}}}}), flush=True)
gate = Path({str(gate_file)!r})
deadline = time.time() + 10
while not gate.exists() and time.time() < deadline:
    time.sleep(0.05)
print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "TASK_DONE:" + prompt}}}}), flush=True)
print(json.dumps({{"type": "turn.completed"}}), flush=True)
""",
    )

    with with_fake_codex(stream_script):
        create_res = send_req(
            "tools/call",
            {
                "name": "codex",
                "arguments": {"prompt": "run tests", "model": "gpt-6-astra"},
                **tasks_meta,
            },
        ).responses()[0]["result"]

    assert create_res["resultType"] == "task"
    task_id = create_res["taskId"]
    assert re.match(r"^[0-9a-f]{32}$", task_id), f"expected 32-char hex taskId, got {task_id}"
    assert create_res["status"] == "working"
    assert create_res["ttlMs"] == m._TASK_MAX_AGE * 1000
    assert create_res["pollIntervalMs"] == 2000
    assert create_res["createdAt"].endswith("Z")
    assert create_res["lastUpdatedAt"].endswith("Z")

    # Wait briefly for JSONL activity to flush, then poll tasks/get while working
    working_res = None
    for _ in range(40):
        poll = send_req("tasks/get", {"taskId": task_id, **tasks_meta}).responses()[0]["result"]
        if "exec_command" in poll.get("statusMessage", ""):
            working_res = poll
            break
        time.sleep(0.05)
    assert working_res is not None, "tasks/get did not surface live JSONL tool activity"
    assert working_res["resultType"] == "complete"
    assert working_res["status"] == "working"
    assert "pytest -q" in working_res["statusMessage"]
    assert working_res["pollIntervalMs"] == 2000

    # Release gate so the task finishes cleanly
    gate_file.write_text("go")
    final_res = None
    for _ in range(60):
        poll = send_req("tasks/get", {"taskId": task_id, **tasks_meta}).responses()[0]["result"]
        if poll["status"] != "working":
            final_res = poll
            break
        time.sleep(0.05)
    assert final_res is not None and final_res["status"] == "completed"
    assert final_res["resultType"] == "complete"
    assert final_res["result"]["isError"] is False
    assert "TASK_DONE:run tests" in final_res["result"]["content"][0]["text"]
    assert final_res["result"]["structuredContent"]["thread_id"] == "t-mcp-001"
    assert final_res["result"]["structuredContent"]["model"] == "gpt-6-astra"

    # tasks/cancel on already-completed task MUST return -32602
    cancel_done = send_req("tasks/cancel", {"taskId": task_id, **tasks_meta}).responses()[0]
    assert "error" in cancel_done
    assert cancel_done["error"]["code"] == -32602

    # --- B. codex_reply via Tasks extension + failure terminal result ---
    fail_script = make_fake_codex_script(
        "fake_task_fail.py",
        """
import json, sys
_ = sys.stdin.read()
print(json.dumps({"type": "thread.started", "thread_id": "t-mcp-fail"}), flush=True)
print(json.dumps({"type": "turn.started"}), flush=True)
print(json.dumps({"type": "turn.failed", "error": {"message": "Rate limit exceeded (429)"}}), flush=True)
sys.exit(1)
""",
    )
    with with_fake_codex(fail_script):
        reply_task = send_req(
            "tools/call",
            {
                "name": "codex_reply",
                "arguments": {"threadId": "t-mcp-fail", "prompt": "continue", "model": "gpt-6-astra"},
                **tasks_meta,
            },
        ).responses()[0]["result"]
    assert reply_task["resultType"] == "task"
    fail_tid = reply_task["taskId"]
    assert len(fail_tid) == 32

    fail_poll = None
    for _ in range(60):
        poll = send_req("tasks/get", {"taskId": fail_tid, **tasks_meta}).responses()[0]["result"]
        if poll["status"] != "working":
            fail_poll = poll
            break
        time.sleep(0.05)
    # Per SEP-2663: tool-level failures (isError: true) return status="completed"
    # with the CallToolResult carrying isError: true and structured failure details.
    assert fail_poll is not None and fail_poll["status"] == "completed"
    assert fail_poll["resultType"] == "complete"
    assert "error" not in fail_poll
    assert "Rate limit exceeded" in fail_poll["statusMessage"]
    assert fail_poll["result"]["isError"] is True
    assert fail_poll["result"]["structuredContent"]["status"] == "failed"
    assert fail_poll["result"]["structuredContent"]["failure"]["category"] == "rate_limit"

    # tasks/cancel on already-failed (terminal) task -> -32602
    cancel_failed = send_req("tasks/cancel", {"taskId": fail_tid, **tasks_meta}).responses()[0]
    assert cancel_failed["error"]["code"] == -32602

    # --- C. tasks/cancel on a running task (even without _meta version) + re-cancel returns -32602 ---
    sleep_script = make_fake_codex_script(
        "fake_sleep.py",
        "import time\ntime.sleep(60)\n",
    )
    with with_fake_codex(sleep_script):
        running_task = send_req(
            "tools/call",
            {"name": "codex", "arguments": {"prompt": "slow"}, **tasks_meta},
        ).responses()[0]["result"]
    cancel_tid = running_task["taskId"]

    # Deliberately omit _meta protocolVersion on tasks/cancel to verify resultType: "complete" is always set
    cancel_ok = send_req("tasks/cancel", {"taskId": cancel_tid}).responses()[0]
    assert "result" in cancel_ok
    assert cancel_ok["result"].get("resultType") == "complete"

    after_cancel = send_req("tasks/get", {"taskId": cancel_tid, **tasks_meta}).responses()[0]["result"]
    assert after_cancel["resultType"] == "complete"
    assert after_cancel["status"] == "cancelled"
    assert "result" not in after_cancel
    assert "error" not in after_cancel
    assert "cancelled" in after_cancel["statusMessage"].lower()

    # Cancelling again returns -32602
    cancel_again = send_req("tasks/cancel", {"taskId": cancel_tid, **tasks_meta}).responses()[0]
    assert cancel_again["error"]["code"] == -32602

    # Unknown / invalid taskId on tasks/get & tasks/cancel -> -32602
    assert send_req("tasks/get", {"taskId": "not-valid-id"}).responses()[0]["error"]["code"] == -32602
    assert send_req("tasks/get", {"taskId": "0" * 32}).responses()[0]["error"]["code"] == -32602
    assert send_req("tasks/cancel", {"taskId": "0" * 32}).responses()[0]["error"]["code"] == -32602


# ===================================================================
# 4. Performance: Single-Pass JSONL, Incremental Cache, Zero-Read Poll
# ===================================================================

def test_single_pass_and_incremental_jsonl_cache():
    # Single-pass check on small file
    small_path = m.TASK_DIR / "perf0001.stdout"
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "t-perf"}),
        json.dumps({"type": "turn.started"}),
        json.dumps({"type": "item.started", "item": {"type": "command_execution", "command": "ls -la"}}),
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}}),
    ]
    small_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    real_loads = json.loads
    call_count = 0

    def counting_loads(*a, **kw):
        nonlocal call_count
        call_count += 1
        return real_loads(*a, **kw)

    m.json.loads = counting_loads
    try:
        st = m._parse_jsonl_status(small_path)
    finally:
        m.json.loads = real_loads
    assert st["tools_called"] == 1 and st["last_tool"] == "exec_command"
    assert call_count == len(lines), f"expected single-pass ({len(lines)} json.loads), got {call_count}"

    # Incremental cache on large (>8 KiB) growing file including partial line split
    big_path = m.TASK_DIR / "perf0002.stdout"
    base_lines = [json.dumps({"type": "thread.started", "thread_id": "t-big"})]
    for i in range(120):
        base_lines.append(
            json.dumps({
                "type": "item.started",
                "item": {"type": "command_execution", "command": f"echo step-{i} " + ("x" * 80)},
            })
        )
    big_path.write_text("\n".join(base_lines) + "\n", encoding="utf-8")
    assert big_path.stat().st_size > m._JSONL_CACHE_MIN_BYTES

    # First read populates cache
    a1, s1 = m._parse_jsonl_file(big_path)
    assert s1["tools_called"] == 120 and a1["thread_id"] == "t-big"

    # Second read of unchanged file must do ZERO json.loads
    call_count = 0
    m.json.loads = counting_loads
    try:
        _, s2 = m._parse_jsonl_file(big_path)
    finally:
        m.json.loads = real_loads
    assert s2["tools_called"] == 120
    assert call_count == 0, f"expected cache hit (0 json.loads), got {call_count}"

    # Append a half-written JSON line (no newline yet)
    part1 = '{"type": "item.started", "item": {"type": "command_execution", '
    part2 = '"command": "pytest final"}}\n'
    with open(big_path, "a", encoding="utf-8") as f:
        f.write(part1)
    _, s_partial = m._parse_jsonl_file(big_path)
    assert s_partial["tools_called"] == 120

    # Complete that line and verify ONLY 1 new line is parsed (not all 121!)
    with open(big_path, "a", encoding="utf-8") as f:
        f.write(part2)
    call_count = 0
    m.json.loads = counting_loads
    try:
        _, s3 = m._parse_jsonl_file(big_path)
    finally:
        m.json.loads = real_loads
    assert s3["tools_called"] == 121
    assert s3["last_tool_args"] == "pytest final"
    assert call_count == 1, f"expected incremental parse of 1 line, got {call_count}"

    # Overwrite/truncate in-place (same inode) to <= 8 KiB, then regrow beyond
    # the original size: cache MUST have been evicted on truncation so stale
    # state is never spliced with the new content!
    with open(big_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "thread.started", "thread_id": "t-reset"}) + "\n")
    a4, s4 = m._parse_jsonl_file(big_path)
    assert a4["thread_id"] == "t-reset" and s4["tools_called"] == 0

    regrown_lines = [json.dumps({"type": "thread.started", "thread_id": "t-regrown"})]
    for i in range(130):
        regrown_lines.append(
            json.dumps({
                "type": "item.started",
                "item": {"type": "command_execution", "command": f"echo new-{i} " + ("y" * 90)},
            })
        )
    with open(big_path, "w", encoding="utf-8") as f:
        f.write("\n".join(regrown_lines) + "\n")
    a5, s5 = m._parse_jsonl_file(big_path)
    assert a5["thread_id"] == "t-regrown", f"stale cache leaked old thread_id: {a5['thread_id']}"
    assert s5["tools_called"] == 130, f"stale cache corrupted tools_called: {s5['tools_called']}"


def test_wait_tasks_no_stdout_reads_while_polling_and_enriched_progress():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    tid = "ccdd0001"
    meta_path = m.TASK_DIR / f"{tid}.meta"
    stdout_path = m.TASK_DIR / f"{tid}.stdout"
    meta_path.write_text(json.dumps({
        "task_id": tid,
        "status": "running",
        "pid": child.pid,
        "pid_start_time": m._get_pid_start_time(child.pid),
        "started_at": time.time(),
    }))
    stdout_path.write_text(
        "\n".join([
            json.dumps({"type": "thread.started", "thread_id": "t-prog"}),
            json.dumps({"type": "turn.started"}),
            json.dumps({"type": "item.started", "item": {"type": "command_execution", "command": "cargo test"}}),
        ]) + "\n"
    )

    real_safe_read = m._safe_read
    real_interval = m._PROGRESS_INTERVAL
    stdout_reads = 0

    def spy_safe_read(path):
        nonlocal stdout_reads
        if str(path).endswith(f"{tid}.stdout"):
            stdout_reads += 1
        return real_safe_read(path)

    try:
        # 1. With no progressToken, _wait_tasks must read .stdout ZERO times while pending
        m._safe_read = spy_safe_read
        m._PROGRESS_INTERVAL = 1
        res = m._wait_tasks([tid], timeout=2.5, request_id=None, progress_token=None)
        assert res[tid]["status"] == "timeout"
        assert res[tid]["elapsed_seconds"] >= 0
        assert stdout_reads == 0, f"expected 0 .stdout reads without progressToken, got {stdout_reads}"

        # 2. With progressToken, progress notification is enriched with live JSONL tool activity
        m._safe_read = real_safe_read
        with Captured() as cap:
            m._wait_tasks([tid], timeout=2.5, request_id=None, progress_token="tok-1")
        prog = cap.progress()
        assert len(prog) >= 1
        msg = prog[0]["params"]["message"]
        assert "1 Codex task running" in msg
        assert "exec_command(cargo test)" in msg, f"progress not enriched: {msg!r}"
    finally:
        m._safe_read = real_safe_read
        m._PROGRESS_INTERVAL = real_interval
        child.kill()
        child.wait()


# ===================================================================
# 5. Model Parsing, Prompt Stdin (-), and Argv Safety
# ===================================================================

def test_model_from_command_ignores_prompt_and_unmatched_quotes():
    assert m._model_from_command("codex exec --json -m gpt-6-astra -") == "gpt-6-astra"
    assert m._model_from_command("codex exec --json -c model=gpt-6-astra prompt") == "gpt-6-astra"
    # Prompt containing -m bogus MUST NOT be parsed as the model!
    assert m._model_from_command("codex exec --json fix -m bogus-model") is None
    assert m._model_from_command("codex exec --json 'explain -m fake --model nope'") is None
    # Unmatched quote in legacy command must still find real -m and ignore prompt -m
    assert m._model_from_command("codex exec --json -m real-model it's broken -m bogus") == "real-model"
    assert m._model_from_command("codex exec --json it's broken -m bogus") is None
    # Stops at --
    assert m._model_from_command(["codex", "exec", "--json", "-m", "real-model", "--", "-m", "bogus"]) == "real-model"
    assert m._model_from_command(["codex", "exec", "--json", "--", "-m", "bogus"]) is None
    # Resume subcommand
    assert m._model_from_command("codex exec resume --json -m res-model 019c-uuid -m bogus") == "res-model"


def test_prompt_passed_via_stdin_handles_leading_dash_and_large_payload():
    echo_script = make_fake_codex_script(
        "fake_stdin_echo.py",
        """
import json, sys
argv = sys.argv[1:]
prompt = sys.stdin.read()
assert "-" in argv, f"expected '-' in argv: {argv}"
# Ensure raw prompt text is NOT in argv
for a in argv:
    assert "SECRET_MARKER" not in a, f"prompt leaked to argv: {a[:40]}"
print(json.dumps({"type": "thread.started", "thread_id": "t-stdin"}))
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": f"LEN={len(prompt)}|HEAD={prompt[:25]}|TAIL={prompt[-10:]}"}}))
print(json.dumps({"type": "turn.completed"}))
""",
    )

    # Leading-dash prompt + subcommand-like text + >40KB payload (exceeds Windows 32,767 cap)
    huge_prompt = "-m evil-model --help\nresume\nSECRET_MARKER\n" + ("日本語テスト-" * 6000) + "END_MARKER"
    assert len(huge_prompt) > 40000

    with with_fake_codex(echo_script):
        # 1. Sync codex
        res = send_req(
            "tools/call",
            {"name": "codex", "arguments": {"prompt": huge_prompt, "model": "real-model"}},
        ).responses()[0]["result"]
        assert res["isError"] is False
        text = res["content"][0]["text"]
        assert f"LEN={len(huge_prompt)}" in text
        assert "HEAD=-m evil-model --help" in text
        assert "TAIL=END_MARKER" in text

        # 2. Async codex_async + codex_wait
        launch = send_req(
            "tools/call",
            {"name": "codex_async", "arguments": {"prompt": huge_prompt, "model": "real-model"}},
        ).responses()[0]["result"]
        tid = launch["structuredContent"]["task_id"]
        # Verify stored .meta records resolved model directly and does not embed raw prompt
        meta = json.loads((m.TASK_DIR / f"{tid}.meta").read_text(encoding="utf-8"))
        assert meta["model"] == "real-model"
        assert "SECRET_MARKER" not in meta["command"]

        waited = send_req(
            "tools/call",
            {"name": "codex_wait", "arguments": {"task_ids": [tid], "timeout": 10}},
        ).responses()[0]["result"]
        assert waited["isError"] is False
        assert f"LEN={len(huge_prompt)}" in waited["content"][0]["text"]

        # 3. codex_reply with cwd + model + config + leading-dash threadId
        reply_cmd, reply_cwd = m._build_command({
            "threadId": "-weird-id",
            "prompt": "-m fake",
            "model": "gpt-6-astra",
            "cwd": TEST_ROOT,
            "sandbox": "workspace-write",
            "config": {"foo": "bar"},
        })
        assert "--" in reply_cmd
        assert reply_cmd[reply_cmd.index("--") + 1] == "-weird-id"
        assert reply_cmd[-1] == "-"
        assert reply_cwd == os.path.abspath(TEST_ROOT)
        assert "-s" not in reply_cmd and "sandbox_mode=workspace-write" in reply_cmd


# ===================================================================
# 6. POSIX Process Group Kill (No Orphaned Grandchildren)
# ===================================================================

def test_posix_process_group_kill_on_cancel_and_stop_proc():
    if sys.platform == "win32":
        return

    # Guard against signalling invalid PID or own process group
    assert m._posix_signal_tree(0, signal.SIGTERM) is False
    assert m._posix_signal_tree(-1, signal.SIGTERM) is False

    # --- A. _cancel_task kills leader AND grandchild in same process group ---
    pid_file = Path(TEST_ROOT) / "grandchild_cancel.pid"
    spawner = make_fake_codex_script(
        "fake_group_spawner.py",
        f"""
import subprocess, sys, time
from pathlib import Path
gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path({str(pid_file)!r}).write_text(str(gc.pid))
time.sleep(60)
""",
    )
    with with_fake_codex(spawner):
        launch = m._start_async({"prompt": "spawn"})
    tid = launch["task_id"]
    meta = json.loads((m.TASK_DIR / f"{tid}.meta").read_text(encoding="utf-8"))
    leader_pid = meta["pid"]
    assert meta.get("own_process_group") is True

    # Wait until grandchild PID file is written
    for _ in range(50):
        if pid_file.exists():
            break
        time.sleep(0.05)
    assert pid_file.exists(), "grandchild did not start"
    gc_pid = int(pid_file.read_text().strip())
    assert m._is_alive(gc_pid)

    cancel_res = m._cancel_task(tid)
    assert cancel_res["status"] == "cancelled"

    # Both leader and grandchild must be dead!
    for _ in range(40):
        if not m._is_alive(leader_pid) and not m._is_alive(gc_pid):
            break
        time.sleep(0.05)
    assert not m._is_alive(leader_pid), "leader survived cancel"
    assert not m._is_alive(gc_pid), f"grandchild PID {gc_pid} was orphaned on POSIX cancel!"

    # --- B. _stop_proc kills sync child AND its grandchild ---
    pid_file2 = Path(TEST_ROOT) / "grandchild_stop.pid"
    spawner2 = make_fake_codex_script(
        "fake_group_spawner2.py",
        f"""
import subprocess, sys, time
from pathlib import Path
gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path({str(pid_file2)!r}).write_text(str(gc.pid))
time.sleep(60)
""",
    )
    proc = subprocess.Popen(
        [sys.executable, str(spawner2)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **m._popen_kw(new_group=True),
    )
    for _ in range(50):
        if pid_file2.exists():
            break
        time.sleep(0.05)
    assert pid_file2.exists()
    gc_pid2 = int(pid_file2.read_text().strip())
    assert m._is_alive(gc_pid2)

    m._stop_proc(proc)
    for _ in range(40):
        if not m._is_alive(gc_pid2):
            break
        time.sleep(0.05)
    assert not m._is_alive(gc_pid2), f"grandchild PID {gc_pid2} survived _stop_proc!"


# ===================================================================
# 7. Cancelled codex_wait KeyError Fix
# ===================================================================

def test_codex_wait_cancelled_no_keyerror():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    tid = "eeff0001"
    (m.TASK_DIR / f"{tid}.meta").write_text(json.dumps({
        "task_id": tid,
        "status": "running",
        "pid": child.pid,
        "pid_start_time": m._get_pid_start_time(child.pid),
        "started_at": time.time() - 5,
    }))
    (m.TASK_DIR / f"{tid}.stdout").write_text("")

    rid = _next_id()
    exc_box = []
    captured_frames = []

    def run_wait():
        try:
            with Captured() as cap:
                m._handle({
                    "jsonrpc": "2.0",
                    "id": rid,
                    "method": "tools/call",
                    "params": {"name": "codex_wait", "arguments": {"task_ids": [tid], "timeout": 30}},
                })
                captured_frames.extend(cap.frames)
        except Exception as exc:
            exc_box.append(exc)

    try:
        t = threading.Thread(target=run_wait)
        t.start()
        time.sleep(0.2)
        # Cancel the in-flight codex_wait request
        m._handle({
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": rid},
        })
        t.join(timeout=5)
        assert not t.is_alive(), "codex_wait thread hung after cancellation"
        assert not exc_box, f"codex_wait raised exception on cancel: {exc_box!r}"
        assert captured_frames, "Captured._send saw formatted response without KeyError"
        text = captured_frames[0]["result"]["content"][0]["text"]
        assert "Wait cancelled by client" in text
    finally:
        with m._cancelled_lock:
            m._cancelled_requests.discard(rid)
        child.kill()
        child.wait()


# ===================================================================
# 8. SIGCHLD Signal Safety & Atomic Metadata Concurrency
# ===================================================================

def test_sigchld_handler_has_no_logging_calls():
    src = inspect.getsource(m._sigchld_handler)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "logging":
            raise AssertionError("_sigchld_handler must not reference logging (non-reentrant in signal handlers)")


def test_cancel_preserves_already_exited_task_verdict():
    """Calling codex_cancel on a dead-but-unpolled task must finalize its real
    status instead of overwriting a completed run as cancelled.
    """
    tid = "aabb0099"
    (m.TASK_DIR / f"{tid}.meta").write_text(json.dumps({
        "task_id": tid,
        "status": "running",
        "pid": os.getpid(),
        "pid_start_time": 1.0,  # forces _is_alive -> False
        "started_at": time.time() - 10,
    }))
    (m.TASK_DIR / f"{tid}.stdout").write_text(
        "\n".join([
            json.dumps({"type": "thread.started", "thread_id": "t-finished"}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Already done!"}}),
            json.dumps({"type": "turn.completed"}),
        ]) + "\n"
    )
    (m.TASK_DIR / f"{tid}.stderr").write_text("")

    res = m._cancel_task(tid)
    assert res["status"] == "completed"
    assert res.get("already_terminal") is True
    persisted = json.loads((m.TASK_DIR / f"{tid}.meta").read_text(encoding="utf-8"))
    assert persisted["status"] == "completed"


def test_atomic_meta_writes_under_concurrency():
    tid = "cc001122"
    meta_file = m.TASK_DIR / f"{tid}.meta"
    m._write_meta_atomic(meta_file, {"task_id": tid, "status": "running", "started_at": time.time()})

    errors = []

    def writer(idx):
        try:
            for i in range(40):
                m._write_meta_atomic(
                    meta_file,
                    {"task_id": tid, "status": "running", "started_at": time.time(), "seq": f"{idx}-{i}"},
                )
        except Exception as exc:
            errors.append(exc)

    def reader():
        try:
            for _ in range(80):
                parsed = json.loads(meta_file.read_text(encoding="utf-8"))
                assert parsed["task_id"] == tid
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors, f"concurrent atomic meta errors: {errors!r}"
    # No leftover .tmp files
    assert list(m.TASK_DIR.glob("*.tmp")) == []


# ===================================================================
# 9. Input Validation Edge Cases
# ===================================================================

def test_input_validation_edge_cases():
    # params: null and non-dict params
    res = send_req("ping", None).responses()[0]
    assert "result" in res
    with Captured() as cap:
        m._handle({"jsonrpc": "2.0", "id": _next_id(), "method": "tools/list", "params": []})
    assert cap.responses()[0]["error"]["code"] == -32602

    # arguments: null (previously crashed with AttributeError!)
    err = send_req("tools/call", {"name": "codex", "arguments": None}).responses()[0]["error"]
    assert err["code"] == -32602

    # arguments: non-dict
    err = send_req("tools/call", {"name": "codex", "arguments": "bad"}).responses()[0]["error"]
    assert err["code"] == -32602

    # task_ids passed as string instead of list (previously iterated chars!)
    for tool in ("codex_status", "codex_wait"):
        err = send_req(
            "tools/call", {"name": tool, "arguments": {"task_ids": "deadbeef"}}
        ).responses()[0]["error"]
        assert err["code"] == -32602, f"{tool} accepted string task_ids"

        err2 = send_req(
            "tools/call", {"name": tool, "arguments": {"task_ids": []}}
        ).responses()[0]["error"]
        assert err2["code"] == -32602

        err3 = send_req(
            "tools/call", {"name": tool, "arguments": {"task_ids": [123]}}
        ).responses()[0]["error"]
        assert err3["code"] == -32602

    # Invalid timeout values on codex_wait
    for bad_timeout in (-1, 0, True, "60", float("nan"), float("inf")):
        err = send_req(
            "tools/call",
            {"name": "codex_wait", "arguments": {"task_ids": ["deadbeef"], "timeout": bad_timeout}},
        ).responses()[0]["error"]
        assert err["code"] == -32602, f"codex_wait accepted bad timeout {bad_timeout!r}"

    # Invalid config / worktree / model types
    for bad_args in [
        {"prompt": "hi", "config": "not-a-dict"},
        {"prompt": "hi", "worktree": "yes"},
        {"prompt": "hi", "model": 123},
        {"prompt": "hi", "threadId": 456},
        {"prompt": ""},
    ]:
        err = send_req("tools/call", {"name": "codex", "arguments": bad_args}).responses()[0]["error"]
        assert err["code"] == -32602, f"codex accepted bad args {bad_args!r}"

    # codex_cancel with missing/non-string task_id
    for bad_tid in (None, "", 123):
        err = send_req(
            "tools/call", {"name": "codex_cancel", "arguments": {"task_id": bad_tid}}
        ).responses()[0]["error"]
        assert err["code"] == -32602


def test_terminate_all_sync_procs_kills_posix_process_group():
    if sys.platform == "win32":
        return

    pid_file = Path(TEST_ROOT) / "grandchild_shutdown.pid"
    spawner = make_fake_codex_script(
        "fake_shutdown_spawner.py",
        f"""
import subprocess, sys, time
from pathlib import Path
gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path({str(pid_file)!r}).write_text(str(gc.pid))
time.sleep(60)
""",
    )
    proc = subprocess.Popen(
        [sys.executable, str(spawner)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **m._popen_kw(new_group=True),
    )
    rid = _next_id()
    m._register_sync_proc(rid, proc)
    try:
        for _ in range(50):
            if pid_file.exists():
                break
            time.sleep(0.05)
        assert pid_file.exists()
        gc_pid = int(pid_file.read_text().strip())
        assert m._is_alive(gc_pid)

        m._terminate_all_sync_procs()
        proc.wait(timeout=5)
        for _ in range(40):
            if not m._is_alive(gc_pid):
                break
            time.sleep(0.05)
        assert not m._is_alive(gc_pid), f"grandchild {gc_pid} survived _terminate_all_sync_procs!"
    finally:
        m._unregister_sync_proc(rid)
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_codex_reply_applies_cwd_model_sandbox_and_config():
    sub_dir = Path(TEST_ROOT) / "reply_subdir"
    sub_dir.mkdir(exist_ok=True)
    probe_script = make_fake_codex_script(
        "fake_reply_probe.py",
        """
import json, os, sys
prompt = sys.stdin.read()
payload = {"cwd": os.path.realpath(os.getcwd()), "argv": sys.argv[1:], "prompt": prompt}
print(json.dumps({"type": "thread.started", "thread_id": "t-reply-99"}))
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(payload)}}))
print(json.dumps({"type": "turn.completed"}))
""",
    )

    with with_fake_codex(probe_script):
        res = send_req(
            "tools/call",
            {
                "name": "codex_reply",
                "arguments": {
                    "threadId": "t-reply-99",
                    "prompt": "-continue here",
                    "cwd": str(sub_dir),
                    "model": "custom-reply-model",
                    "sandbox": "read-only",
                    "approval-policy": "never",
                    "config": {"extra_flag": True},
                },
            },
        ).responses()[0]["result"]
    assert res["isError"] is False
    parsed = json.loads(res["content"][0]["text"])
    assert parsed["cwd"] == os.path.realpath(str(sub_dir))
    assert parsed["prompt"] == "-continue here"
    argv = parsed["argv"]
    assert argv[:3] == ["exec", "resume", "--json"]
    assert "-m" in argv and argv[argv.index("-m") + 1] == "custom-reply-model"
    assert "sandbox_mode=read-only" in argv
    assert "approval_policy=never" in argv
    assert "extra_flag=true" in argv
    assert argv[-2:] == ["t-reply-99", "-"]
    assert res["structuredContent"]["model"] == "custom-reply-model"


def test_stdio_server_end_to_end_protocol_lines():
    server_py = Path(__file__).resolve().parent / "codex_mcp_swarm.py"
    env = dict(os.environ)
    env["CODEX_SWARM_TASK_DIR"] = os.path.join(TEST_ROOT, "e2e_tasks")
    env["CODEX_SWARM_WORKTREE_DIR"] = os.path.join(TEST_ROOT, "e2e_worktrees")
    env["CODEX_SWARM_LOG"] = os.path.join(TEST_ROOT, "e2e_server.log")

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": {"_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "2025-11-25", "capabilities": {}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "method": "notifications/unknown_event", "params": {"foo": "bar"}},
        {"jsonrpc": "2.0", "id": None, "method": "notifications/another_unknown"},
        {"jsonrpc": "2.0", "id": 3, "method": "ping"},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": {"_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "ping", "params": {"_meta": {"io.modelcontextprotocol/protocolVersion": "2099-01-01"}}},
    ]
    stdin_payload = "\n".join(json.dumps(r) for r in requests) + "\n"
    proc = subprocess.run(
        [sys.executable, str(server_py), "-c", "model=gpt-6-astra"],
        input=stdin_payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0, f"server exited {proc.returncode}: {proc.stderr}"
    frames = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    # Only the 5 requests with non-null IDs should produce responses (zero for the 3 notifications!)
    assert [f["id"] for f in frames] == [1, 2, 3, 4, 5], f"unexpected response ids: {frames}"
    assert frames[0]["result"]["resultType"] == "complete"
    assert "io.modelcontextprotocol/tasks" in frames[0]["result"]["capabilities"]["extensions"]
    assert frames[1]["result"]["protocolVersion"] == "2025-11-25"
    assert frames[2]["result"] == {}
    assert frames[3]["result"]["resultType"] == "complete"
    assert len(frames[3]["result"]["tools"]) == 6
    assert frames[4]["error"]["code"] == -32022


def test_malformed_jsonrpc_unhashable_id_method_and_cancel_request_id():
    """Malformed JSON-RPC frames with unhashable id/method/requestId must return
    -32600 (or be silently ignored for notifications) and NEVER crash the server.
    """
    server_py = Path(__file__).resolve().parent / "codex_mcp_swarm.py"
    env = dict(os.environ)
    env["CODEX_SWARM_TASK_DIR"] = os.path.join(TEST_ROOT, "malformed_tasks")
    env["CODEX_SWARM_WORKTREE_DIR"] = os.path.join(TEST_ROOT, "malformed_worktrees")
    env["CODEX_SWARM_LOG"] = os.path.join(TEST_ROOT, "malformed_server.log")

    requests = [
        {"jsonrpc": "2.0", "id": [1], "method": "ping"},
        {"jsonrpc": "2.0", "id": {"bad": True}, "method": "ping"},
        {"jsonrpc": "2.0", "id": True, "method": "ping"},
        {"jsonrpc": "2.0", "id": 10, "method": ["tools/call"]},
        {"jsonrpc": "2.0", "id": 11},
        {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": [1, 2]}},
        {"jsonrpc": "2.0", "id": 12, "method": "ping"},
    ]
    stdin_payload = "\n".join(json.dumps(r) for r in requests) + "\n"
    proc = subprocess.run(
        [sys.executable, str(server_py)],
        input=stdin_payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0, f"server crashed on malformed input: rc={proc.returncode} stderr={proc.stderr}"
    frames = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    # Expect 5 -32600 errors + 1 valid ping response (id=12), zero response for notification
    assert len(frames) == 6, f"unexpected frames: {frames}"
    for f in frames[:3]:
        assert f["id"] is None and f["error"]["code"] == -32600
    assert frames[3]["id"] == 10 and frames[3]["error"]["code"] == -32600
    assert frames[4]["id"] == 11 and frames[4]["error"]["code"] == -32600
    assert frames[5]["id"] == 12 and frames[5]["result"] == {}


def test_cancel_task_sigkill_reaps_zombie():
    """Cancelling a child that ignores SIGTERM (requiring SIGKILL) must reap it
    so it never lingers as a zombie process.
    """
    if sys.platform == "win32":
        return

    ign_script = make_fake_codex_script(
        "fake_ignore_sigterm.py",
        "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)\n",
    )
    with with_fake_codex(ign_script):
        launch = m._start_async({"prompt": "stubborn"})
    tid = launch["task_id"]
    pid = json.loads((m.TASK_DIR / f"{tid}.meta").read_text(encoding="utf-8"))["pid"]
    time.sleep(0.1)

    res = m._cancel_task(tid)
    assert res["status"] == "cancelled"
    assert pid not in m._ASYNC_PIDS and pid not in m._ASYNC_PROCS

    # On Linux, /proc/<pid> disappears once reaped (a zombie keeps /proc/<pid>/status with State: Z)
    proc_status = Path(f"/proc/{pid}/status")
    for _ in range(20):
        if not proc_status.exists():
            break
        time.sleep(0.05)
    assert not proc_status.exists(), f"PID {pid} lingered as zombie: {proc_status.read_text()}"


def test_toml_quoted_server_config_stripped_for_flags():
    """SERVER_CONFIG values passed in TOML-quoted `-c key=\"val\"` syntax must
    have quotes stripped when lifted to `-m` or `-s` CLI flags.
    """
    saved = dict(m.SERVER_CONFIG)
    m.SERVER_CONFIG.clear()
    m.SERVER_CONFIG["model"] = '"gpt-6-astra"'
    m.SERVER_CONFIG["sandbox_mode"] = '"danger-full-access"'
    try:
        cmd, _ = m._build_command({"prompt": "hello"})
        assert cmd[cmd.index("-m") + 1] == "gpt-6-astra"
        assert cmd[cmd.index("-s") + 1] == "danger-full-access"
        assert m._resolve_model({}) == "gpt-6-astra"
    finally:
        m.SERVER_CONFIG.clear()
        m.SERVER_CONFIG.update(saved)


def test_all_tools_structured_content_matches_output_schema():
    """Validate structuredContent from ALL 6 tools against their strict outputSchema."""
    tools_by_name = {t["name"]: t for t in m.TOOLS}

    ok_script = make_fake_codex_script(
        "fake_schema_ok.py",
        """
import json, sys
_ = sys.stdin.read()
print(json.dumps({"type": "thread.started", "thread_id": "t-schema-1"}))
print(json.dumps({"type": "turn.started"}))
print(json.dumps({"type": "item.started", "item": {"type": "command_execution", "command": "echo hi"}}))
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Schema OK"}}))
print(json.dumps({"type": "turn.completed"}))
""",
    )
    sleep_script = make_fake_codex_script(
        "fake_schema_sleep.py",
        "import time\ntime.sleep(60)\n",
    )

    with with_fake_codex(ok_script):
        # 1. codex_async launch
        a_res = send_req(
            "tools/call",
            {"name": "codex_async", "arguments": {"prompt": "hi", "model": "gpt-6-astra"}},
        ).responses()[0]["result"]
        assert a_res["isError"] is False
        validate_against_schema(a_res["structuredContent"], tools_by_name["codex_async"]["outputSchema"])
        tid = a_res["structuredContent"]["task_id"]

        # 2. codex_wait completed
        w_res = send_req(
            "tools/call",
            {"name": "codex_wait", "arguments": {"task_ids": [tid], "timeout": 10}},
        ).responses()[0]["result"]
        assert w_res["isError"] is False
        validate_against_schema(w_res["structuredContent"], tools_by_name["codex_wait"]["outputSchema"])

        # 3. codex_status completed
        s_res = send_req(
            "tools/call",
            {"name": "codex_status", "arguments": {"task_ids": [tid]}},
        ).responses()[0]["result"]
        assert s_res["isError"] is False
        validate_against_schema(s_res["structuredContent"], tools_by_name["codex_status"]["outputSchema"])

        # 4. codex_cancel already completed
        c_res = send_req(
            "tools/call",
            {"name": "codex_cancel", "arguments": {"task_id": tid}},
        ).responses()[0]["result"]
        assert c_res["isError"] is False
        validate_against_schema(c_res["structuredContent"], tools_by_name["codex_cancel"]["outputSchema"])

        # 5. codex_reply success
        r_res = send_req(
            "tools/call",
            {"name": "codex_reply", "arguments": {"threadId": "t-schema-1", "prompt": "more"}},
        ).responses()[0]["result"]
        assert r_res["isError"] is False
        validate_against_schema(r_res["structuredContent"], tools_by_name["codex_reply"]["outputSchema"])

    # 6. codex_wait timeout + codex_cancel running
    real_floor = m._WAIT_MIN_TIMEOUT
    m._WAIT_MIN_TIMEOUT = 1
    try:
        with with_fake_codex(sleep_script):
            a2 = send_req(
                "tools/call",
                {"name": "codex_async", "arguments": {"prompt": "slow"}},
            ).responses()[0]["result"]
            tid2 = a2["structuredContent"]["task_id"]
            w2 = send_req(
                "tools/call",
                {"name": "codex_wait", "arguments": {"task_ids": [tid2], "timeout": 1}},
            ).responses()[0]["result"]
            assert w2["isError"] is False
            validate_against_schema(w2["structuredContent"], tools_by_name["codex_wait"]["outputSchema"])
            assert w2["structuredContent"]["tasks"][0]["status"] == "timeout"

            c2 = send_req(
                "tools/call",
                {"name": "codex_cancel", "arguments": {"task_id": tid2}},
            ).responses()[0]["result"]
            assert c2["isError"] is False
            validate_against_schema(c2["structuredContent"], tools_by_name["codex_cancel"]["outputSchema"])
            assert c2["structuredContent"]["status"] == "cancelled"
    finally:
        m._WAIT_MIN_TIMEOUT = real_floor


def test_real_git_worktree_head_only_and_cleanup():
    """Exercise real git worktree creation, verify HEAD-only exclusion of
    uncommitted changes, verify spawn failure cleans up the unlaunched worktree,
    and verify _remove_worktree cleans up directory and branch.
    """
    repo = Path(TEST_ROOT) / "wt_repo"
    repo.mkdir()
    git_env = dict(os.environ)
    git_env["GIT_AUTHOR_NAME"] = "Test"
    git_env["GIT_AUTHOR_EMAIL"] = "test@example.com"
    git_env["GIT_COMMITTER_NAME"] = "Test"
    git_env["GIT_COMMITTER_EMAIL"] = "test@example.com"
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=git_env)
    (repo / "committed.txt").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "committed.txt"], check=True, env=git_env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True, env=git_env)
    # Create uncommitted modification + untracked file
    (repo / "committed.txt").write_text("uncommitted_edit\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("secret_untracked\n", encoding="utf-8")

    probe_wt = make_fake_codex_script(
        "fake_wt_probe.py",
        """
import json, os, sys
from pathlib import Path
_ = sys.stdin.read()
cwd = Path(os.getcwd())
info = {
    "committed": (cwd / "committed.txt").read_text().strip(),
    "has_untracked": (cwd / "untracked.txt").exists(),
}
print(json.dumps({"type": "thread.started", "thread_id": "t-wt"}))
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(info)}}))
print(json.dumps({"type": "turn.completed"}))
""",
    )
    with with_fake_codex(probe_wt):
        res = send_req(
            "tools/call",
            {"name": "codex", "arguments": {"prompt": "check wt", "cwd": str(repo), "worktree": True}},
        ).responses()[0]["result"]
    assert res["isError"] is False
    sc = res["structuredContent"]
    assert sc["worktree_branch"].startswith("codex-swarm/")
    wt_path = Path(sc["worktree_path"])
    assert wt_path.exists()
    # Extract JSON payload before the --- footer
    payload = json.loads(res["content"][0]["text"].split("\n\n---\n")[0])
    assert payload["committed"] == "v1", "worktree should reflect HEAD, not uncommitted edits"
    assert payload["has_untracked"] is False, "worktree should not include untracked files"

    # Cleanup removes both worktree dir and branch
    assert m._remove_worktree(str(wt_path), sc["worktree_branch"]) is True
    assert not wt_path.exists()
    branches = subprocess.run(
        ["git", "-C", str(repo), "branch", "--list", sc["worktree_branch"]],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()
    assert branches == "", f"branch survived cleanup: {branches}"

    # Spawn failure with worktree=True must clean up the unlaunched worktree
    real_exec = m._exec_argv
    m._exec_argv = lambda cmd: ["/nonexistent_codex_binary_12345"]
    try:
        err_res = send_req(
            "tools/call",
            {"name": "codex", "arguments": {"prompt": "fail spawn", "cwd": str(repo), "worktree": True}},
        ).responses()[0]["result"]
        assert err_res["isError"] is True
        assert "worktree_path" not in err_res["structuredContent"]

        err_async = send_req(
            "tools/call",
            {"name": "codex_async", "arguments": {"prompt": "fail spawn", "cwd": str(repo), "worktree": True}},
        ).responses()[0]["result"]
        assert err_async["isError"] is True
    finally:
        m._exec_argv = real_exec

    # No leftover codex-swarm/* branches after failed launches
    leftover = subprocess.run(
        ["git", "-C", str(repo), "branch", "--list", "codex-swarm/*"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()
    assert leftover == "", f"unlaunched worktree branch leaked: {leftover}"


def test_concurrent_cancel_vs_resolve_and_tasks_get():
    """Race _cancel_task against _resolve_task_state and tasks/get on real
    short-lived processes exiting right as cancellation is invoked.
    """
    quick_script = make_fake_codex_script(
        "fake_quick_race.py",
        """
import json, sys, time
_ = sys.stdin.read()
time.sleep(0.08)
print(json.dumps({"type": "thread.started", "thread_id": "t-race"}), flush=True)
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "finished"}}), flush=True)
print(json.dumps({"type": "turn.completed"}), flush=True)
""",
    )
    errors = []
    with with_fake_codex(quick_script):
        for _ in range(5):
            launch = m._start_async_internal(
                {"prompt": "race"}, high_entropy_id=True, mcp_task=True, tool_name="codex"
            )
            tid = launch["task_id"]
            time.sleep(0.05)

            def do_cancel(t=tid):
                try:
                    m._cancel_task(t)
                except Exception as exc:
                    errors.append(exc)

            def do_resolve(t=tid):
                try:
                    for _ in range(5):
                        m._resolve_task_state(t)
                        time.sleep(0.01)
                except Exception as exc:
                    errors.append(exc)

            def do_get(t=tid):
                try:
                    for _ in range(5):
                        send_req("tasks/get", {"taskId": t})
                        time.sleep(0.01)
                except Exception as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=do_cancel),
                threading.Thread(target=do_resolve),
                threading.Thread(target=do_get),
            ]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=5)
            final = m._resolve_task_state(tid)
            assert final["status"] in ("completed", "cancelled", "failed"), f"unexpected status: {final}"
    assert not errors, f"race errors: {errors!r}"


def test_check_task_avoids_redundant_stdout_read_for_jsonl():
    """Once _parse_jsonl_file parses JSONL with a structured result or error,
    _check_task must not call _safe_read on .stdout again.
    """
    tid = "f00d0001"
    (m.TASK_DIR / f"{tid}.meta").write_text(json.dumps({
        "task_id": tid,
        "status": "completed",
        "started_at": time.time() - 5,
        "completed_at": time.time(),
        "exit_code": 0,
    }))
    (m.TASK_DIR / f"{tid}.stdout").write_text(
        "\n".join([
            json.dumps({"type": "thread.started", "thread_id": "t-noread"}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Cached answer"}}),
            json.dumps({"type": "turn.completed"}),
        ]) + "\n"
    )
    (m.TASK_DIR / f"{tid}.stderr").write_text("")

    reads = []
    real_safe_read = m._safe_read

    def spy(path):
        reads.append(str(path))
        return real_safe_read(path)

    m._safe_read = spy
    try:
        info = m._check_task(tid)
    finally:
        m._safe_read = real_safe_read
    assert info["result"] == "Cached answer"
    assert not any(r.endswith(f"{tid}.stdout") for r in reads), f"redundant .stdout read: {reads}"


def test_jsonl_incremental_shrink_race_and_file_not_found_invalidation():
    """Verify two incremental JSONL cache race fixes:
    1. If a file shrinks between stat() and open() to a size between
       entry['offset'] and entry['size'] on the same inode, the incremental
       branch MUST NOT splice stale state and must fall back to a full read.
    2. If the file disappears between stat() and open() (FileNotFoundError),
       the cache entry must be evicted and empty state returned.
    """
    import builtins

    # --- 1. Slight shrink race (entry["offset"] <= shrunk_size < entry["size"]) ---
    shrink_path = m.TASK_DIR / "race_shrink.stdout"
    base_lines = [json.dumps({"type": "thread.started", "thread_id": "t-orig"})]
    for i in range(120):
        base_lines.append(
            json.dumps({
                "type": "item.started",
                "item": {"type": "command_execution", "command": f"echo orig-{i} " + ("a" * 80)},
            })
        )
    # Trailing partial line (no newline) makes entry["size"] > entry["offset"] by 200+ bytes
    partial_tail = '{"type": "item.started", "item": {"type": "command_execution", "command": "' + ("p" * 180)
    shrink_path.write_text("\n".join(base_lines) + "\n" + partial_tail, encoding="utf-8")
    assert shrink_path.stat().st_size > m._JSONL_CACHE_MIN_BYTES

    a1, s1 = m._parse_jsonl_file(shrink_path)
    assert a1["thread_id"] == "t-orig" and s1["tools_called"] == 120
    cached = m._JSONL_CACHE[str(shrink_path)]
    orig_offset = cached["offset"]
    orig_size = cached["size"]
    assert orig_size - orig_offset > 150

    # Grow file so initial stat() enters the incremental branch
    with open(shrink_path, "a", encoding="utf-8") as f:
        f.write('" + "z" * 100"}}\n' + (" " * 200) + "\n")
    assert shrink_path.stat().st_size > orig_size

    # Build fresh replacement content whose byte size lands strictly in [orig_offset, orig_size)
    fresh_lines = [json.dumps({"type": "thread.started", "thread_id": "t-shrunk-fresh"})]
    for i in range(5):
        fresh_lines.append(
            json.dumps({
                "type": "item.started",
                "item": {"type": "command_execution", "command": f"echo fresh-{i}"},
            })
        )
    fresh_text = "\n".join(fresh_lines) + "\n"
    pad_needed = (orig_offset + 50) - len(fresh_text.encode("utf-8"))
    assert pad_needed > 0
    # Pad with blank lines so byte length is orig_offset + 50 (same inode, >= offset, < size)
    fresh_text += "\n" * pad_needed
    assert orig_offset <= len(fresh_text.encode("utf-8")) < orig_size

    real_open = builtins.open
    triggered = {"done": False}

    def shrink_on_first_open(file, mode="r", *args, **kwargs):
        if Path(file) == shrink_path and mode == "rb" and not triggered["done"]:
            triggered["done"] = True
            with real_open(shrink_path, "w", encoding="utf-8") as wf:
                wf.write(fresh_text)
        return real_open(file, mode, *args, **kwargs)

    m.open = shrink_on_first_open
    try:
        a2, s2 = m._parse_jsonl_file(shrink_path)
    finally:
        delattr(m, "open")

    assert triggered["done"] is True
    assert a2["thread_id"] == "t-shrunk-fresh", f"stale state spliced after shrink race: {a2}"
    assert s2["tools_called"] == 5, f"stale tool count leaked: {s2}"

    # --- 2. FileNotFoundError race inside incremental branch ---
    vanish_path = m.TASK_DIR / "race_vanish.stdout"
    vanish_path.write_text("\n".join(base_lines) + "\n", encoding="utf-8")
    assert vanish_path.stat().st_size > m._JSONL_CACHE_MIN_BYTES
    a_v1, s_v1 = m._parse_jsonl_file(vanish_path)
    assert a_v1["thread_id"] == "t-orig" and s_v1["tools_called"] == 120
    assert str(vanish_path) in m._JSONL_CACHE

    # Grow file so stat() triggers incremental branch, then unlink right before open()
    with open(vanish_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "turn.completed"}) + "\n")

    def vanish_on_open(file, mode="r", *args, **kwargs):
        if Path(file) == vanish_path and mode == "rb":
            vanish_path.unlink(missing_ok=True)
        return real_open(file, mode, *args, **kwargs)

    m.open = vanish_on_open
    try:
        a_v2, s_v2 = m._parse_jsonl_file(vanish_path)
    finally:
        delattr(m, "open")

    assert str(vanish_path) not in m._JSONL_CACHE, "cache entry not invalidated on FileNotFoundError"
    assert a_v2["saw_json"] is False and a_v2["thread_id"] is None
    assert s_v2["tools_called"] == 0


def test_prompt_stdin_posix_and_windows_regular_file_lifecycle():
    """Test both _prompt_stdin branches:
    - POSIX: unlinked TemporaryFile (no .stdin file created on disk).
    - Windows (mocked _IS_WINDOWS=True): regular .stdin file (no O_TEMPORARY)
      cleaned up after sync process exit / launch failure, and for async tasks
      cleaned up on launch failure, finalization, cancellation, and old-task cleanup.
    """
    sample_prompt = "-m tricky\nUTF-8 café 日本語"

    # --- A. Direct _prompt_stdin checks on POSIX vs Windows ---
    saved_win = m._IS_WINDOWS
    try:
        # 1. Empty / None yields DEVNULL on both platforms
        for win_flag in (False, True):
            m._IS_WINDOWS = win_flag
            for empty in ("", None):
                with m._prompt_stdin(empty, stdin_path=m.TASK_DIR / "unused.stdin") as fh:
                    assert fh == subprocess.DEVNULL
                assert not (m.TASK_DIR / "unused.stdin").exists()

        # 2. POSIX branch: uses unlinked TemporaryFile, never creates stdin_path
        m._IS_WINDOWS = False
        posix_path = m.TASK_DIR / "posix_no_file.stdin"
        before_sync_files = set(m.TASK_DIR.glob("sync_*.stdin"))
        with m._prompt_stdin(sample_prompt, stdin_path=posix_path) as fh:
            assert not posix_path.exists()
            assert fh.read() == sample_prompt.encode("utf-8")
        assert set(m.TASK_DIR.glob("sync_*.stdin")) == before_sync_files

        # 3. Windows branch (sync, stdin_path=None): regular sync_*.stdin file exists
        #    while context is active (even after fh.close()) and is removed on exit
        m._IS_WINDOWS = True
        with m._prompt_stdin(sample_prompt) as fh:
            created = set(m.TASK_DIR.glob("sync_*.stdin")) - before_sync_files
            assert len(created) == 1, f"expected 1 sync .stdin file, got {created}"
            sync_stdin = next(iter(created))
            assert sync_stdin.read_bytes() == sample_prompt.encode("utf-8")
            assert fh.read() == sample_prompt.encode("utf-8")
            fh.close()
            # File remains on disk while child would still be running
            assert sync_stdin.exists()
        assert not sync_stdin.exists(), "sync .stdin file leaked after context exit"

        # Windows sync exception path also cleans up sync_*.stdin
        try:
            with m._prompt_stdin(sample_prompt) as fh:
                created = set(m.TASK_DIR.glob("sync_*.stdin")) - before_sync_files
                assert len(created) == 1
                err_stdin = next(iter(created))
                raise RuntimeError("simulated Popen failure")
        except RuntimeError:
            pass
        assert not err_stdin.exists(), "sync .stdin file leaked on exception"

        # 4. Windows branch (async, stdin_path provided): regular file persists after
        #    context exit so detached child/grandchild keeps reading it
        async_stdin = m.TASK_DIR / "winasync01.stdin"
        with m._prompt_stdin(sample_prompt, stdin_path=async_stdin) as fh:
            assert async_stdin.exists()
            assert fh.read() == sample_prompt.encode("utf-8")
        assert async_stdin.exists(), "async .stdin file was prematurely unlinked"
        async_stdin.unlink(missing_ok=True)
    finally:
        m._IS_WINDOWS = saved_win

    # --- B. End-to-end mocked Windows sync & async lifecycle with real subprocesses ---
    win_probe_script = make_fake_codex_script(
        "fake_win_stdin_probe.py",
        f"""
import json, sys
from pathlib import Path
task_dir = Path({str(m.TASK_DIR)!r})
sync_files = [p.name for p in task_dir.glob("sync_*.stdin")]
prompt = sys.stdin.read()
print(json.dumps({{"type": "thread.started", "thread_id": "t-win-stdin"}}), flush=True)
print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": json.dumps({{"prompt": prompt, "sync_files": len(sync_files)}})}}}}), flush=True)
print(json.dumps({{"type": "turn.completed"}}), flush=True)
""",
    )
    sleep_script = make_fake_codex_script(
        "fake_win_sleep.py",
        "import sys, time\n_ = sys.stdin.read()\ntime.sleep(60)\n",
    )

    real_is_windows = m._IS_WINDOWS
    real_popen_kw = m._popen_kw
    real_get_start = m._get_pid_start_time
    real_is_alive = m._is_alive
    real_win_kill = getattr(m, "_win_kill_tree", None)
    real_last_cleanup = m._last_cleanup

    def posix_popen_kw(**_kwargs):
        return {"start_new_session": True} if sys.platform != "win32" else {}

    def posix_is_alive(pid, expected_start_time=None):
        if not pid or pid <= 0:
            return False
        proc = m._ASYNC_PROCS.get(pid)
        if proc is not None and proc.poll() is not None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def fake_win_kill_tree(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        return True

    try:
        m._IS_WINDOWS = True
        m._popen_kw = posix_popen_kw
        m._get_pid_start_time = lambda pid: None
        m._is_alive = posix_is_alive
        m._win_kill_tree = fake_win_kill_tree

        # 1. Sync codex & codex_reply: .stdin file exists during child run, cleaned after exit
        with with_fake_codex(win_probe_script):
            out_sync = m._run_sync_detailed({"prompt": sample_prompt})
            assert out_sync["status"] == "completed", out_sync
            parsed_sync = json.loads(out_sync["text"])
            assert parsed_sync["prompt"] == sample_prompt
            assert parsed_sync["sync_files"] >= 1, "expected regular sync_*.stdin on disk during child run"
            assert list(m.TASK_DIR.glob("sync_*.stdin")) == []

            out_reply = m._run_reply_sync_detailed("t-win-stdin", sample_prompt)
            assert out_reply["status"] == "completed", out_reply
            parsed_reply = json.loads(out_reply["text"])
            assert parsed_reply["prompt"] == sample_prompt
            assert parsed_reply["sync_files"] >= 1
            assert list(m.TASK_DIR.glob("sync_*.stdin")) == []

        # 2. Async launch failure: cleans up .stdout, .stderr, AND .stdin
        real_exec = m._exec_argv
        m._exec_argv = lambda cmd: ["/nonexistent_codex_win_test_12345"]
        before_files = set(m.TASK_DIR.iterdir())
        try:
            try:
                m._start_async({"prompt": sample_prompt})
                raise AssertionError("expected launch failure")
            except OSError:
                pass
        finally:
            m._exec_argv = real_exec
        assert set(m.TASK_DIR.iterdir()) == before_files, "async launch failure leaked task files"

        # 3. Async normal completion: .stdin exists at launch, child reads it, removed on finalization
        with with_fake_codex(win_probe_script):
            launched = m._start_async({"prompt": sample_prompt})
        tid = launched["task_id"]
        # Wait until process exits and finalize
        for _ in range(60):
            st = m._resolve_task_state(tid)
            if st["status"] != "running":
                break
            time.sleep(0.05)
        assert st["status"] == "completed", st
        assert not (m.TASK_DIR / f"{tid}.stdin").exists(), ".stdin not cleaned up on finalization"
        checked = m._check_task(tid)
        assert json.loads(checked["result"])["prompt"] == sample_prompt

        # 4. Async cancel: .stdin exists while running, removed on _cancel_task
        with with_fake_codex(sleep_script):
            launched2 = m._start_async({"prompt": sample_prompt})
        tid2 = launched2["task_id"]
        assert (m.TASK_DIR / f"{tid2}.stdin").exists(), "expected .stdin while Windows async task runs"
        cancel_res = m._cancel_task(tid2)
        assert cancel_res["status"] == "cancelled", cancel_res
        assert not (m.TASK_DIR / f"{tid2}.stdin").exists(), ".stdin not cleaned up on cancel"

        # 5. _cleanup_old_tasks removes leftover .stdin AND old orphaned sync_*.stdin
        old_tid = "dead0123"
        old_ts = time.time() - (m._TASK_MAX_AGE + 3600)
        (m.TASK_DIR / f"{old_tid}.meta").write_text(
            json.dumps({"task_id": old_tid, "status": "completed", "started_at": old_ts, "completed_at": old_ts}),
            encoding="utf-8",
        )
        for ext in (".stdout", ".stderr", ".stdin"):
            (m.TASK_DIR / f"{old_tid}{ext}").write_text("old", encoding="utf-8")
        orphan_sync = m.TASK_DIR / "sync_orphan_test.stdin"
        orphan_sync.write_text("stale", encoding="utf-8")
        os.utime(orphan_sync, (old_ts, old_ts))

        m._last_cleanup = 0.0
        m._cleanup_old_tasks()
        for ext in (".meta", ".stdout", ".stderr", ".stdin"):
            assert not (m.TASK_DIR / f"{old_tid}{ext}").exists(), f"{ext} not cleaned by _cleanup_old_tasks"
        assert not orphan_sync.exists(), "orphan sync_*.stdin not cleaned by _cleanup_old_tasks"
    finally:
        m._IS_WINDOWS = real_is_windows
        m._popen_kw = real_popen_kw
        m._get_pid_start_time = real_get_start
        m._is_alive = real_is_alive
        if real_win_kill is None:
            if hasattr(m, "_win_kill_tree"):
                delattr(m, "_win_kill_tree")
        else:
            m._win_kill_tree = real_win_kill
        m._last_cleanup = real_last_cleanup


def test_jsonl_unicode_line_separators_preserved():
    """RFC 8259 allows raw U+2028/U+2029/U+0085 inside JSON strings (Rust
    serde_json emits them unescaped). Neither single-pass nor incremental JSONL
    parsing may split events on those characters.
    """
    msg = "alpha\u2028beta\u2029gamma\x85delta"
    small_jsonl = (
        json.dumps({"type": "thread.started", "thread_id": "t-ls-small"}, ensure_ascii=False)
        + "\n"
        + json.dumps(
            {"type": "item.completed", "item": {"type": "agent_message", "text": msg}},
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps({"type": "turn.completed"}, ensure_ascii=False)
        + "\n"
    )
    a_small, s_small = m._parse_jsonl_full(small_jsonl)
    assert a_small["thread_id"] == "t-ls-small"
    assert a_small["last_assistant_text"] == msg
    assert a_small["terminal_status"] == "completed"
    assert s_small["last_assistant_text"] == msg

    # Large file (>8 KiB) + incremental append containing U+2028/U+2029
    big_path = m.TASK_DIR / "unicode_ls_big.stdout"
    lines = [json.dumps({"type": "thread.started", "thread_id": "t-ls-big"}, ensure_ascii=False)]
    for i in range(120):
        lines.append(
            json.dumps(
                {
                    "type": "item.started",
                    "item": {"type": "command_execution", "command": f"echo {i} " + ("x" * 80)},
                },
                ensure_ascii=False,
            )
        )
    big_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert big_path.stat().st_size > m._JSONL_CACHE_MIN_BYTES
    a1, s1 = m._parse_jsonl_file(big_path)
    assert a1["thread_id"] == "t-ls-big" and s1["tools_called"] == 120

    with open(big_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"type": "item.completed", "item": {"type": "agent_message", "text": msg}},
                ensure_ascii=False,
            )
            + "\n"
            + json.dumps({"type": "turn.completed"}, ensure_ascii=False)
            + "\n"
        )
    a2, s2 = m._parse_jsonl_file(big_path)
    assert a2["last_assistant_text"] == msg
    assert a2["terminal_status"] == "completed"
    assert s2["phase"] == "completed"


def test_jsonl_rollout_non_string_fields_no_crash_and_schema_valid():
    """Malformed/non-string rollout response_item fields must never raise
    TypeError or violate codex_status outputSchema string types.
    """
    tools_by_name = {t["name"]: t for t in m.TOOLS}
    cases = [
        json.dumps({"type": "thread.started", "thread_id": 12345}),
        json.dumps({"type": "response_item", "payload": {"type": "function_call", "name": "shell", "arguments": {"cmd": ["ls", "-la"]}}}),
        json.dumps({"type": "response_item", "payload": {"type": "function_call", "name": "shell", "arguments": {"cmd": 42}}}),
        json.dumps({"type": "response_item", "payload": {"type": "function_call", "name": None, "arguments": "{}"}}),
        json.dumps({"type": "response_item", "payload": {"type": "reasoning", "content": 999}}),
        json.dumps({"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": 123}}),
        json.dumps({"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": None}, {"type": "output_text", "text": "safe"}]}}),
    ]
    for line in cases:
        analysis, status = m._parse_jsonl_full(line + "\n")
        assert analysis["thread_id"] is None or isinstance(analysis["thread_id"], str)
        if status["last_tool"] is not None:
            assert isinstance(status["last_tool"], str)
        if status["last_tool_args"] is not None:
            assert isinstance(status["last_tool_args"], str)
        m._format_task_activity_brief(status)

    tid = "badd0001"
    (m.TASK_DIR / f"{tid}.meta").write_text(
        json.dumps({"task_id": tid, "status": "completed", "started_at": time.time() - 5, "completed_at": time.time(), "exit_code": 0}),
        encoding="utf-8",
    )
    (m.TASK_DIR / f"{tid}.stdout").write_text("\n".join(cases) + "\n", encoding="utf-8")
    (m.TASK_DIR / f"{tid}.stderr").write_text("", encoding="utf-8")

    res = send_req("tools/call", {"name": "codex_status", "arguments": {"task_ids": [tid]}}).responses()[0]["result"]
    assert res["isError"] is False
    validate_against_schema(res["structuredContent"], tools_by_name["codex_status"]["outputSchema"])


def test_sync_notifications_cancelled_status_pre_spawn_and_nonblocking():
    """Verify sync cancellation fixes:
    1. Cancelling an in-flight sync call via notifications/cancelled reports
       status='cancelled' (not 'failed' exit -15).
    2. notifications/cancelled does not block _handle() on a child that ignores
       SIGTERM (termination runs off the main stdin thread).
    3. Pre-cancelled requests never spawn Codex or create a worktree.
    """
    # --- 1. Pre-cancelled request must not spawn Codex ---
    ran_marker = Path(TEST_ROOT) / "pre_cancel_ran"
    marker_script = make_fake_codex_script(
        "fake_pre_cancel.py",
        f"from pathlib import Path\nPath({str(ran_marker)!r}).write_text('RAN')\n",
    )
    pre_rid = _next_id()
    m._mark_cancelled(pre_rid)
    try:
        with with_fake_codex(marker_script):
            out = m._run_sync_detailed({"prompt": "never run"}, request_id=pre_rid)
            out_reply = m._run_reply_sync_detailed("t-pre", "never run", request_id=pre_rid)
        assert out["status"] == "cancelled" and out["failed"] is True
        assert out_reply["status"] == "cancelled" and out_reply["failed"] is True
        assert not ran_marker.exists(), "pre-cancelled request still spawned codex"
    finally:
        with m._cancelled_lock:
            m._cancelled_requests.discard(pre_rid)

    # --- 2. In-flight sync cancel reports status='cancelled' and doesn't block _handle ---
    if sys.platform == "win32":
        return

    stubborn_script = make_fake_codex_script(
        "fake_sync_cancel_stubborn.py",
        "import signal, sys, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n_ = sys.stdin.read()\ntime.sleep(60)\n",
    )
    rid = _next_id()
    box = {}

    def run_sync():
        with with_fake_codex(stubborn_script):
            box["outcome"] = m._run_sync_detailed({"prompt": "cancel me"}, request_id=rid)

    t = threading.Thread(target=run_sync)
    t.start()
    try:
        for _ in range(100):
            with m._sync_procs_lock:
                if rid in m._SYNC_PROCS:
                    break
            time.sleep(0.02)
        with m._sync_procs_lock:
            assert rid in m._SYNC_PROCS, "sync proc did not register"

        # notifications/cancelled must return immediately (not block 5s waiting for SIGKILL!)
        t0 = time.time()
        m._handle({
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": rid},
        })
        elapsed_handle = time.time() - t0
        assert elapsed_handle < 1.0, f"notifications/cancelled blocked main thread for {elapsed_handle:.2f}s"

        t.join(timeout=10)
        assert not t.is_alive(), "cancelled sync thread hung"
        assert box["outcome"]["status"] == "cancelled", f"expected status='cancelled', got {box['outcome']}"
        assert box["outcome"]["failed"] is True
    finally:
        with m._cancelled_lock:
            m._cancelled_requests.discard(rid)


def test_jsonl_incremental_exact_size_shrink_and_zero_inode():
    """Verify exact-size shrink race (shrunk_size == entry['size']) falls back
    to full read instead of splicing stale state, and st_ino == 0 bypasses cache.
    """
    import builtins

    p = m.TASK_DIR / "race_shrink_exact.stdout"
    base_lines = [json.dumps({"type": "thread.started", "thread_id": "t-exact-old"})]
    for i in range(120):
        base_lines.append(
            json.dumps({
                "type": "item.started",
                "item": {"type": "command_execution", "command": f"echo old-{i} " + ("x" * 80)},
            })
        )
    partial = '{"type": "item.started", "item": {"type": "command_execution", "command": "' + ("p" * 200)
    p.write_text("\n".join(base_lines) + "\n" + partial, encoding="utf-8")

    a1, s1 = m._parse_jsonl_file(p)
    assert a1["thread_id"] == "t-exact-old" and s1["tools_called"] == 120
    orig_size = m._JSONL_CACHE[str(p)]["size"]

    # Grow past orig_size so stat() enters the incremental branch
    with open(p, "a", encoding="utf-8") as f:
        f.write('zzz"}}\n' + (" " * 100) + "\n")

    # Replace in-place with fresh content of exact byte size == orig_size
    fresh = json.dumps({"type": "thread.started", "thread_id": "t-exact-new"}) + "\n"
    fresh += "\n" * (orig_size - len(fresh.encode("utf-8")))
    assert len(fresh.encode("utf-8")) == orig_size

    real_open = builtins.open
    done = [False]

    def shrink_exact_on_open(file, mode="r", *args, **kwargs):
        if Path(file) == p and mode == "rb" and not done[0]:
            done[0] = True
            with real_open(p, "w", encoding="utf-8") as wf:
                wf.write(fresh)
        return real_open(file, mode, *args, **kwargs)

    m.open = shrink_exact_on_open
    try:
        a2, s2 = m._parse_jsonl_file(p)
    finally:
        delattr(m, "open")

    assert done[0] is True
    assert a2["thread_id"] == "t-exact-new", f"stale state spliced on exact-size shrink: {a2}"
    assert s2["tools_called"] == 0, f"stale tool count leaked on exact-size shrink: {s2}"

    # Verify st_ino == 0 (e.g., FAT32/SMB/FUSE) bypasses inode-based caching cleanly
    p_zero = m.TASK_DIR / "zero_ino.stdout"
    p_zero.write_text("\n".join(base_lines) + "\n", encoding="utf-8")
    assert p_zero.stat().st_size > m._JSONL_CACHE_MIN_BYTES
    real_stat = Path.stat

    class _ZeroInoStat:
        def __init__(self, st):
            self._st = st
            self.st_ino = 0

        def __getattr__(self, name):
            return getattr(self._st, name)

    def fake_zero_ino_stat(self, *args, **kwargs):
        st = real_stat(self, *args, **kwargs)
        return _ZeroInoStat(st) if self == p_zero else st

    Path.stat = fake_zero_ino_stat
    try:
        az, sz = m._parse_jsonl_file(p_zero)
    finally:
        Path.stat = real_stat
    assert az["thread_id"] == "t-exact-old" and sz["tools_called"] == 120
    assert str(p_zero) not in m._JSONL_CACHE, "st_ino == 0 file must not be cached by inode"


def test_start_async_meta_write_failure_cleans_up_child_files_and_tracking():
    """If _write_meta_atomic fails after Popen in _start_async_internal, the
    spawned child must be killed, _ASYNC_PIDS/_ASYNC_PROCS cleaned, and task
    files (.stdout/.stderr/.stdin) unlinked (including Windows .stdin path).
    """
    sleep_script = make_fake_codex_script(
        "fake_meta_fail_sleep.py",
        "import sys, time\n_ = sys.stdin.read()\ntime.sleep(60)\n",
    )
    real_write_meta = m._write_meta_atomic
    real_is_windows = m._IS_WINDOWS
    real_popen_kw = m._popen_kw
    real_get_start = m._get_pid_start_time
    real_win_kill = getattr(m, "_win_kill_tree", None)

    for win_mode in (False, True):
        before_files = set(m.TASK_DIR.iterdir())
        before_pids = set(m._ASYNC_PIDS)
        spawned_pids = []

        def failing_write_meta(meta_file, meta, _box=spawned_pids):
            _box.append(meta["pid"])
            raise OSError("simulated disk full")

        try:
            m._write_meta_atomic = failing_write_meta
            if win_mode:
                m._IS_WINDOWS = True
                m._popen_kw = lambda **_kw: {"start_new_session": True} if sys.platform != "win32" else {}
                m._get_pid_start_time = lambda pid: None
                m._win_kill_tree = lambda pid: (os.kill(pid, signal.SIGKILL) or True) if pid else True
            with with_fake_codex(sleep_script):
                try:
                    m._start_async({"prompt": "will fail meta write"})
                    raise AssertionError("expected OSError from _write_meta_atomic")
                except OSError:
                    pass
        finally:
            m._write_meta_atomic = real_write_meta
            m._IS_WINDOWS = real_is_windows
            m._popen_kw = real_popen_kw
            m._get_pid_start_time = real_get_start
            if real_win_kill is None:
                if hasattr(m, "_win_kill_tree"):
                    delattr(m, "_win_kill_tree")
            else:
                m._win_kill_tree = real_win_kill

        assert len(spawned_pids) == 1
        leaked_pid = spawned_pids[0]
        assert not m._is_alive(leaked_pid), f"child PID {leaked_pid} leaked (win_mode={win_mode})"
        assert set(m._ASYNC_PIDS) == before_pids, f"_ASYNC_PIDS leaked {leaked_pid}"
        assert leaked_pid not in m._ASYNC_PROCS, f"_ASYNC_PROCS leaked {leaked_pid}"
        assert set(m.TASK_DIR.iterdir()) == before_files, (
            f"task files leaked after meta write failure (win_mode={win_mode}): "
            f"{set(m.TASK_DIR.iterdir()) - before_files}"
        )


def test_sigchld_handler_does_not_resurrect_popped_pid_and_vanished_meta_not_found():
    """Verify:
    1. _sigchld_handler never resurrects a PID concurrently popped from _ASYNC_PIDS.
    2. A .meta file vanishing between exists() and read_text() yields status='not_found'
       (-32602 on tasks/get), not 'error' (-32603).
    """
    # 1. Simulate worker popping PID right before waitpid raises ChildProcessError
    fake_pid = 99999999
    m._ASYNC_PIDS[fake_pid] = m._NOT_REAPED
    real_waitpid = m.os.waitpid

    def racing_waitpid(pid, flags):
        if pid == fake_pid:
            m._ASYNC_PIDS.pop(fake_pid, None)
            raise ChildProcessError("already reaped by worker")
        return real_waitpid(pid, flags)

    m.os.waitpid = racing_waitpid
    try:
        m._sigchld_handler(None, None)
    finally:
        m.os.waitpid = real_waitpid
    assert fake_pid not in m._ASYNC_PIDS, "_sigchld_handler resurrected a popped PID"

    # 2. Vanished .meta file returns not_found (-32602) instead of error (-32603)
    tid = "dead9999"
    meta_file = m.TASK_DIR / f"{tid}.meta"
    meta_file.write_text(json.dumps({"task_id": tid, "status": "completed"}), encoding="utf-8")

    real_read_text = Path.read_text

    def vanish_read_text(self, *args, **kwargs):
        if self == meta_file:
            meta_file.unlink(missing_ok=True)
            raise FileNotFoundError(str(self))
        return real_read_text(self, *args, **kwargs)

    Path.read_text = vanish_read_text
    try:
        st = m._resolve_task_state(tid)
        assert st["status"] == "not_found", f"expected not_found on vanished meta, got {st}"
        meta_file.write_text(json.dumps({"task_id": tid, "status": "completed"}), encoding="utf-8")
        err = send_req("tasks/get", {"taskId": tid}).responses()[0]["error"]
        assert err["code"] == -32602, f"expected -32602 on vanished meta, got {err}"
    finally:
        Path.read_text = real_read_text
        meta_file.unlink(missing_ok=True)


def _run_all():
    tests = [
        fn for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    passed, failed = 0, []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            failed.append((fn.__name__, exc))
    print(f"\n{passed} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())

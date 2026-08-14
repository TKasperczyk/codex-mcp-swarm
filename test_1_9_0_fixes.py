#!/usr/bin/env python3
"""
Regression tests for the 1.9.0 fixes.

Covers the three things that let a session strand a finished Codex result:
protocol version negotiation, the codex_wait progress heartbeat, and the
codex_wait floor that keeps the call above the client's backgrounding
threshold. Drives the real _handle() with _send captured, so no Codex
process is ever spawned.
"""
import os
import sys
import time
import json
import types
import threading

os.environ.setdefault("CODEX_SWARM_TASK_DIR", "/tmp/codex_swarm_tasks_test")
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
        lock = threading.Lock()

        def fake(resp):
            with lock:
                self.frames.append(resp)

        m._send = fake
        return self

    def __exit__(self, *a):
        m._send = self._real

    def responses(self):
        return [f for f in self.frames if "result" in f or "error" in f]

    def progress(self):
        return [f for f in self.frames if f.get("method") == "notifications/progress"]


def text_of(resp):
    return "\n".join(
        c.get("text", "") for c in resp.get("result", {}).get("content", [])
    )


# ---------------------------------------------------------------------------
print("\nprotocol negotiation")
# ---------------------------------------------------------------------------
for asked, expect, label in [
    ("2025-11-25", "2025-11-25", "echoes the revision Claude Code 2.1.232 asks for"),
    ("2024-11-05", "2024-11-05", "still honours the old revision"),
    ("2099-01-01", m._DEFAULT_PROTOCOL_VERSION, "falls back on an unknown revision"),
]:
    with Captured() as cap:
        m._handle({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": asked, "capabilities": {}},
        })
    got = cap.responses()[0]["result"]["protocolVersion"]
    check(label, got == expect, f"asked {asked}, got {got}")

check(
    "no longer hardcodes 2024-11-05",
    m._DEFAULT_PROTOCOL_VERSION == "2025-11-25",
    m._DEFAULT_PROTOCOL_VERSION,
)

# ---------------------------------------------------------------------------
print("\ncodex_async response text")
# ---------------------------------------------------------------------------
real_start = m._start_async
m._start_async = lambda args: {"task_id": "deadbeef"}
try:
    with Captured() as cap:
        m._handle({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "codex_async", "arguments": {"prompt": "x"}},
        })
    body = text_of(cap.responses()[0])
finally:
    m._start_async = real_start

check("drops the 'started in background' phrasing", "in background" not in body, repr(body[:60]))
check("states the client is not tracking it", "NOT TRACKED BY YOUR CLIENT" in body)
check("names codex_wait as mandatory", 'codex_wait(task_ids=["deadbeef"])' in body)
check("warns the turn must not end first", "before this turn ends" in body)
check("demotes codex_status", "does not collect the result" in body)

# ---------------------------------------------------------------------------
print("\ncodex_wait floor")
# ---------------------------------------------------------------------------
check(
    "default floor clears the 120s backgrounding threshold",
    m._WAIT_MIN_TIMEOUT > m._CLIENT_AUTO_BACKGROUND_S,
    f"{m._WAIT_MIN_TIMEOUT}s vs {m._CLIENT_AUTO_BACKGROUND_S}s",
)

# A finished task must still return instantly: the floor must not turn a
# ready result into a two minute stall.
m.TASK_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
done_id = "aaaa1111"
(m.TASK_DIR / f"{done_id}.meta").write_text(json.dumps({
    "task_id": done_id, "status": "completed", "pid": 999999,
    "started_at": time.time() - 60, "completed_at": time.time() - 1,
    "exit_code": 0, "result": "all done",
}))
(m.TASK_DIR / f"{done_id}.stdout").write_text("")

t0 = time.time()
with Captured() as cap:
    m._handle({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "codex_wait", "arguments": {"task_ids": [done_id], "timeout": 5}},
    })
elapsed = time.time() - t0
check("a finished task returns immediately despite the floor", elapsed < 5, f"{elapsed:.2f}s")
check("and reports completion", "completed in" in text_of(cap.responses()[0]))

# A running task must be held past the floor, with a heartbeat throughout.
# Shrink both knobs so the test takes seconds rather than minutes; the values
# under test are the relationship between them, not the wall clock.
floor = 6
m._WAIT_MIN_TIMEOUT, real_interval = floor, m._PROGRESS_INTERVAL
m._PROGRESS_INTERVAL = 2

child = __import__("subprocess").Popen(["sleep", "600"])
run_id = "bbbb2222"
(m.TASK_DIR / f"{run_id}.meta").write_text(json.dumps({
    "task_id": run_id, "status": "running", "pid": child.pid,
    "pid_start_time": m._get_pid_start_time(child.pid),
    "started_at": time.time(),
}))
(m.TASK_DIR / f"{run_id}.stdout").write_text("")

try:
    t0 = time.time()
    with Captured() as cap:
        m._handle({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {
                "name": "codex_wait",
                "arguments": {"task_ids": [run_id], "timeout": 1},  # asks for 1s
                # Claude Code attaches this to every tools/call; the heartbeat
                # is correlated to it and the spec forbids sending progress
                # without one.
                "_meta": {"progressToken": 4},
            },
        })
    elapsed = time.time() - t0
    progress = cap.progress()

    check(
        "a 1s wait on a running task is raised to the floor",
        elapsed >= floor - 1,
        f"waited {elapsed:.1f}s, floor {floor}s",
    )
    check("progress notifications are emitted during the wait", len(progress) >= 2, f"{len(progress)} sent")
    if progress:
        p = progress[0]["params"]
        check("progress carries the client's token", p.get("progressToken") == 4, str(p.get("progressToken")))
        check("progress message reads as a sentence", "1 Codex task running" in p.get("message", ""), p.get("message"))
    check(
        "the still-running task is reported, not killed",
        "STILL RUNNING" in text_of(cap.responses()[0]) and child.poll() is None,
    )
finally:
    m._WAIT_MIN_TIMEOUT, m._PROGRESS_INTERVAL = m._CLIENT_AUTO_BACKGROUND_S + 30, real_interval
    child.kill()
    child.wait()
    for suffix in (".meta", ".stdout"):
        for tid in (done_id, run_id):
            (m.TASK_DIR / f"{tid}{suffix}").unlink(missing_ok=True)

# ---------------------------------------------------------------------------
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
sys.exit(1 if FAIL else 0)

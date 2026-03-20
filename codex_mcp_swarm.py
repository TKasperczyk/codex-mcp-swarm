#!/usr/bin/env python3
"""
codex-mcp-swarm -- Parallel Codex MCP Server

An MCP server that wraps OpenAI's Codex CLI with true parallel execution,
live task monitoring, and full parameter parity with the official Codex
MCP tool.

Features:
  - codex:        Synchronous execution (drop-in replacement)
  - codex_async:  Fire-and-forget background execution
  - codex_reply:  Continue a previous Codex session
  - codex_status: Live view of what each task is doing
  - codex_wait:   Block until multiple tasks complete

Server-level defaults are set via -c key=value CLI args, matching the
exact same format as `codex mcp-server` for drop-in config compatibility.

Originally inspired by jeanchristophe13v/codex-mcp-async.
Rewritten with full flag parity, live JSONL status parsing, batch wait,
and session reply support.

License: MIT
Repository: https://github.com/TKasperczyk/codex-mcp-swarm
"""

import sys
import json
import subprocess
import uuid
import os
import time
import logging
import signal
import traceback
import argparse
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

__version__ = "1.5.0"

# ---------------------------------------------------------------------------
# Logging (configurable via env vars)
# ---------------------------------------------------------------------------
LOG_FILE = os.environ.get("CODEX_SWARM_LOG", "/tmp/codex_mcp_swarm.log")
LOG_LEVEL = os.environ.get("CODEX_SWARM_LOG_LEVEL", "WARNING").upper()

logging.basicConfig(
    filename=LOG_FILE,
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Task storage
# ---------------------------------------------------------------------------
TASK_DIR = Path(os.environ.get("CODEX_SWARM_TASK_DIR", "/tmp/codex_swarm_tasks"))
TASK_DIR.mkdir(exist_ok=True, mode=0o700)

# ---------------------------------------------------------------------------
# Server-level config (populated in main from CLI args)
# ---------------------------------------------------------------------------
SERVER_CONFIG: Dict[str, str] = {}
SERVER_FLAGS: List[str] = []

# Track async child PIDs so the SIGCHLD handler only reaps those.
# Values: _NOT_REAPED sentinel = not yet reaped; int = real exit code.
# Exit codes: >= 0 for normal exit, negative (-signal) for signal death.
_NOT_REAPED = object()
_ASYNC_PIDS: Dict[int, Any] = {}  # pid -> _NOT_REAPED | int (exit code)
_ASYNC_PROCS: Dict[int, subprocess.Popen] = {}  # pid -> Popen (kept alive for returncode)

# Per-task finalization locks to prevent concurrent _resolve_task_state races
_task_finalize_locks: Dict[str, threading.Lock] = {}
_task_finalize_guard = threading.Lock()


def _get_task_lock(task_id: str) -> threading.Lock:
    with _task_finalize_guard:
        if task_id not in _task_finalize_locks:
            _task_finalize_locks[task_id] = threading.Lock()
        return _task_finalize_locks[task_id]


# ---------------------------------------------------------------------------
# SIGCHLD handler -- reap only tracked async children
# ---------------------------------------------------------------------------
def _sigchld_handler(signum, frame):
    for pid in list(_ASYNC_PIDS):
        if _ASYNC_PIDS.get(pid) is not _NOT_REAPED:
            continue  # already reaped or removed by another thread
        try:
            rpid, status = os.waitpid(pid, os.WNOHANG)
            if rpid == pid:
                if os.WIFEXITED(status):
                    exit_code = os.WEXITSTATUS(status)
                elif os.WIFSIGNALED(status):
                    exit_code = -os.WTERMSIG(status)  # negative = killed by signal
                else:
                    exit_code = 127  # unknown abnormal termination
                _ASYNC_PIDS[pid] = exit_code
                logging.debug("Reaped async child PID %d (exit %d)", pid, exit_code)
        except ChildProcessError:
            # Already reaped by subprocess internals -- exit code lost
            _ASYNC_PIDS[pid] = None
        except Exception as exc:
            logging.warning("SIGCHLD handler error for PID %d: %s", pid, exc)


signal.signal(signal.SIGCHLD, _sigchld_handler)

# ===================================================================
# Utility helpers
# ===================================================================

def _safe_read(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[Error reading {path}: {exc}]"


def _get_pid_start_time(pid: int) -> Optional[float]:
    """Get process start time from /proc (Linux). Returns None if unavailable."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # Field 22 (0-indexed: 21) is starttime in clock ticks
        fields = stat.rsplit(")", 1)[-1].split()
        return float(fields[19])  # index 19 after the closing paren
    except Exception:
        return None


def _is_alive(pid: Optional[int], expected_start_time: Optional[float] = None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False

    # Check for zombie via /proc (Linux)
    try:
        stat_path = Path(f"/proc/{pid}/status")
        if stat_path.exists():
            for line in stat_path.read_text().splitlines():
                if line.startswith("State:"):
                    if "Z" in line:
                        return False
    except Exception:
        pass

    # Guard against PID reuse: if we recorded the start time, verify it matches
    if expected_start_time is not None:
        actual = _get_pid_start_time(pid)
        if actual is not None and actual != expected_start_time:
            return False

    return True


_send_lock = threading.Lock()
_cancelled_requests: Set[Any] = set()
_cancelled_lock = threading.Lock()


def _is_cancelled(rid: Any) -> bool:
    with _cancelled_lock:
        return rid in _cancelled_requests


def _send(response: Dict[str, Any]) -> None:
    # Don't send responses for cancelled requests
    rid = response.get("id")
    if rid is not None and _is_cancelled(rid):
        with _cancelled_lock:
            _cancelled_requests.discard(rid)
        logging.info("Suppressed response for cancelled request id=%s", rid)
        return
    with _send_lock:
        try:
            out = json.dumps(response, ensure_ascii=False)
            print(out, flush=True)
            logging.debug("Sent id=%s (%d bytes)", rid, len(out))
        except (TypeError, ValueError) as exc:
            logging.error("Serialization failed: %s", exc)
            err = {
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": -32603, "message": f"Serialization error: {exc}"},
            }
            print(json.dumps(err), flush=True)


_TASK_ID_RE = re.compile(r"^[0-9a-f]{8}$")


def _validate_task_id(task_id: str) -> bool:
    """Reject task IDs that aren't our generated 8-char hex format."""
    return bool(_TASK_ID_RE.match(task_id))


def _extract_result(stdout: str, stderr: str) -> Tuple[str, Optional[str]]:
    """Extract result and thread ID from codex output. Returns (text, thread_id)."""
    if stdout.strip().startswith("{"):
        text, thread_id = _extract_from_jsonl(stdout)
        if text:
            return text, thread_id
    result = stdout.strip()
    if not result and stderr:
        result = stderr.strip()
    return result or "No output from Codex", None


def _extract_from_jsonl(jsonl_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract the final assistant message and thread ID from JSONL output.
    Handles both `codex exec --json` format (item.completed/agent_message)
    and session file format (response_item/assistant).
    """
    last_assistant_text = None
    thread_id = None
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            etype = event.get("type", "")

            # -- thread/session ID extraction
            if etype == "thread.started":
                thread_id = event.get("thread_id")
            elif etype == "session_meta":
                payload = event.get("payload", {})
                if not thread_id:
                    thread_id = payload.get("id")

            # -- codex exec --json format
            elif etype == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    text = item.get("text", "")
                    if text:
                        last_assistant_text = text

            # -- session file format (fallback)
            elif etype == "response_item":
                payload = event.get("payload", {})
                if payload.get("role") == "assistant" and payload.get("type") == "message":
                    content = payload.get("content") or []
                    texts = [
                        c.get("text", "")
                        for c in content
                        if c.get("type") in ("output_text", "text", "input_text")
                    ]
                    if texts:
                        last_assistant_text = "\n".join(texts)

        except (json.JSONDecodeError, KeyError):
            continue
    return last_assistant_text, thread_id


def _parse_jsonl_status(stdout_path: Path) -> Dict[str, Any]:
    """
    Parse the JSONL stdout file to determine current Codex activity.
    Handles both `codex exec --json` format and session file format.
    """
    status: Dict[str, Any] = {
        "phase": "starting",
        "tools_called": 0,
        "last_tool": None,
        "last_tool_args": None,
        "last_reasoning": None,
        "last_assistant_text": None,
    }

    if not stdout_path.exists():
        return status

    try:
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return status

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        # ---- codex exec --json format ----

        if etype == "turn.started":
            status["phase"] = "running"

        elif etype in ("item.started", "item.completed"):
            item = event.get("item", {})
            itype = item.get("type", "")

            if itype == "command_execution":
                if etype == "item.started":
                    status["tools_called"] += 1
                    status["last_tool"] = "exec_command"
                    status["last_tool_args"] = item.get("command", "")[:150]
                # On completed, capture output
                elif etype == "item.completed":
                    output = item.get("aggregated_output", "")
                    if output:
                        status["last_reasoning"] = output[-200:]

            elif itype == "agent_message" and etype == "item.completed":
                text_val = item.get("text", "")
                if text_val:
                    status["last_assistant_text"] = text_val[-300:]

        # ---- session file format (fallback) ----

        elif etype == "event_msg":
            payload = event.get("payload", {})
            if payload.get("type") == "task_started":
                status["phase"] = "running"

        elif etype == "response_item":
            payload = event.get("payload", {})
            ptype = payload.get("type", "")

            if ptype == "function_call":
                status["tools_called"] += 1
                status["last_tool"] = payload.get("name", "?")
                try:
                    args = json.loads(payload.get("arguments", "{}"))
                    status["last_tool_args"] = (
                        args.get("cmd")
                        or args.get("command")
                        or args.get("path")
                        or args.get("pattern")
                        or str(args)[:150]
                    )
                except (json.JSONDecodeError, TypeError):
                    status["last_tool_args"] = payload.get("arguments", "")[:150]

            elif ptype == "reasoning":
                content = payload.get("content") or []
                for c in content:
                    text_val = c.get("text", "")
                    if text_val:
                        status["last_reasoning"] = text_val[-200:]

            elif payload.get("role") == "assistant" and ptype == "message":
                content = payload.get("content") or []
                for c in content:
                    if c.get("type") in ("output_text", "text", "input_text"):
                        status["last_assistant_text"] = c.get("text", "")[-300:]

    return status


def _flatten_config(prefix: str, value: Any, out: Dict[str, str]) -> None:
    """Flatten nested config values into TOML-compatible dotted key/value pairs."""
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten_config(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, bool):
        out[prefix] = "true" if value else "false"
    elif isinstance(value, list):
        # TOML array syntax with proper string escaping.
        # Nested dicts/lists in arrays are not supported by -c key=value;
        # skip them with a warning.
        parts = []
        for item in value:
            if isinstance(item, (dict, list)):
                logging.warning("Skipping unsupported nested %s in config list %s",
                                type(item).__name__, prefix)
                continue
            if isinstance(item, str):
                escaped = item.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                parts.append(f'"{escaped}"')
            elif isinstance(item, bool):
                parts.append("true" if item else "false")
            else:
                parts.append(str(item))
        out[prefix] = f"[{', '.join(parts)}]"
    else:
        out[prefix] = str(value)

# ===================================================================
# Command builder
# ===================================================================

def _build_command(params: dict) -> Tuple[List[str], Optional[str]]:
    """
    Build a `codex exec` command from tool parameters + server defaults.
    Returns (cmd_list, cwd_or_none).
    """
    cmd = ["codex", "exec"]

    merged = dict(SERVER_CONFIG)
    per_call = params.get("config") or {}
    for k, v in per_call.items():
        _flatten_config(str(k), v, merged)

    # Model
    model = params.get("model")
    if model:
        cmd.extend(["-m", model])
        merged.pop("model", None)
    elif "model" in merged:
        cmd.extend(["-m", merged.pop("model")])

    # Sandbox
    sandbox = params.get("sandbox")
    if sandbox:
        cmd.extend(["-s", sandbox])
        merged.pop("sandbox_mode", None)
    elif "sandbox_mode" in merged:
        cmd.extend(["-s", merged.pop("sandbox_mode")])

    # Approval policy (no dedicated flag -- stays as -c)
    approval = params.get("approval-policy")
    if approval:
        merged["approval_policy"] = approval

    # Profile
    profile = params.get("profile")
    if profile:
        cmd.extend(["-p", profile])

    # CWD
    cwd = params.get("cwd")
    if cwd:
        cmd.extend(["-C", cwd])

    # Text-based params -> config keys
    for param_key, config_key in [
        ("base-instructions", "base_instructions"),
        ("developer-instructions", "developer_instructions"),
        ("compact-prompt", "compact_prompt"),
    ]:
        val = params.get(param_key)
        if val:
            merged[config_key] = val

    for key, value in merged.items():
        cmd.extend(["-c", f"{key}={value}"])

    cmd.extend(SERVER_FLAGS)

    prompt = params.get("prompt", "")
    if prompt:
        cmd.append(prompt)

    proc_cwd = cwd if cwd and os.path.isabs(cwd) else None
    return cmd, proc_cwd


def _build_reply_command(thread_id: str, prompt: str) -> List[str]:
    """Build a `codex exec resume` command for continuing a session."""
    cmd = ["codex", "exec", "resume"]

    model = SERVER_CONFIG.get("model")
    if model:
        cmd.extend(["-m", model])

    for key, value in SERVER_CONFIG.items():
        if key == "model":
            continue
        cmd.extend(["-c", f"{key}={value}"])

    cmd.extend(SERVER_FLAGS)
    cmd.append(thread_id)
    cmd.append(prompt)
    return cmd

# ===================================================================
# Sync execution (cancellable via Popen + poll)
# ===================================================================

_POLL_INTERVAL = 2  # seconds between cancellation/timeout checks


def _wait_proc(
    proc: subprocess.Popen,
    deadline: Optional[float] = None,
    request_id: Any = None,
    timeout_msg: str = "Error: Codex execution timed out",
) -> Tuple[str, Optional[str]]:
    """
    Wait for a Popen process with cancellation and timeout support.
    Shared by sync codex and reply paths.
    """
    while True:
        remaining = None
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return timeout_msg, None
        wait_time = min(_POLL_INTERVAL, remaining) if remaining is not None else _POLL_INTERVAL
        try:
            stdout, stderr = proc.communicate(timeout=wait_time)
            return _extract_result(stdout, stderr)
        except subprocess.TimeoutExpired:
            if request_id is not None and _is_cancelled(request_id):
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return "Cancelled by client", None


def _run_sync(
    params: dict,
    timeout: Optional[int] = None,
    request_id: Any = None,
) -> Tuple[str, Optional[str]]:
    """Run codex synchronously with cancellation support."""
    cmd, cwd = _build_command(params)
    # Add --json for structured output (enables thread_id extraction)
    if "--json" not in cmd:
        cmd.insert(2, "--json")
    logging.info("Sync exec: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )
        deadline = time.time() + timeout if timeout is not None else None
        return _wait_proc(proc, deadline=deadline, request_id=request_id,
                          timeout_msg="Error: Codex execution timed out")
    except Exception as exc:
        return f"Error calling Codex: {exc}", None


def _run_reply_sync(
    thread_id: str,
    prompt: str,
    timeout: Optional[int] = None,
    request_id: Any = None,
) -> Tuple[str, Optional[str]]:
    """Run codex reply synchronously with cancellation support."""
    cmd = _build_reply_command(thread_id, prompt)
    logging.info("Reply exec: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.time() + timeout if timeout is not None else None
        return _wait_proc(proc, deadline=deadline, request_id=request_id,
                          timeout_msg="Error: Codex reply timed out")
    except Exception as exc:
        return f"Error calling Codex reply: {exc}", None

# ===================================================================
# Async execution
# ===================================================================

try:
    _TASK_MAX_AGE = int(os.environ.get("CODEX_SWARM_TASK_MAX_AGE", 86400))
except (ValueError, TypeError):
    _TASK_MAX_AGE = 86400  # default 24h
_last_cleanup = 0.0


def _cleanup_old_tasks() -> None:
    """Remove task artifacts older than _TASK_MAX_AGE seconds."""
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < 300:  # Run at most every 5 minutes
        return
    _last_cleanup = now

    for meta_file in TASK_DIR.glob("*.meta"):
        try:
            meta = json.loads(meta_file.read_text())
            task_id = meta_file.stem
            status = meta.get("status")

            # Finalize unpolled "running" tasks whose process has died
            if status == "running":
                pid = meta.get("pid")
                pid_start_time = meta.get("pid_start_time")
                if _is_alive(pid, expected_start_time=pid_start_time):
                    continue  # still running, skip
                # Dead but never finalized -- resolve it now
                _resolve_task_state(task_id)
                # Re-read metadata after finalization
                meta = json.loads(meta_file.read_text())
                status = meta.get("status")

            if status not in ("completed", "failed"):
                continue
            completed_at = meta.get("completed_at", 0)
            if now - completed_at < _TASK_MAX_AGE:
                continue

            # Delete task files.
            # In-memory tracking (_ASYNC_PIDS/_ASYNC_PROCS) is already cleaned
            # by _resolve_task_state() during finalization -- no need to touch it here.
            for ext in (".meta", ".stdout", ".stderr"):
                (TASK_DIR / f"{task_id}{ext}").unlink(missing_ok=True)

            # Prune per-task finalization lock
            with _task_finalize_guard:
                _task_finalize_locks.pop(task_id, None)

            logging.debug("Cleaned up old task %s", task_id)
        except Exception:
            continue


def _start_async(params: dict) -> str:
    _cleanup_old_tasks()

    task_id = uuid.uuid4().hex[:8]
    cmd, cwd = _build_command(params)
    # Add --json for structured output (enables live status parsing)
    if "--json" not in cmd:
        cmd.insert(2, "--json")
    logging.info("Async start [%s]: %s", task_id, " ".join(cmd))

    stdout_f = TASK_DIR / f"{task_id}.stdout"
    stderr_f = TASK_DIR / f"{task_id}.stderr"

    with open(stdout_f, "w") as out, open(stderr_f, "w") as err:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            text=True,
            cwd=cwd,
            start_new_session=True,
        )

    # Track PID for SIGCHLD handler (sentinel replaced on reap)
    _ASYNC_PIDS[proc.pid] = _NOT_REAPED
    # Keep Popen alive so Python's finalizer doesn't steal the exit code
    _ASYNC_PROCS[proc.pid] = proc

    meta = {
        "task_id": task_id,
        "pid": proc.pid,
        "pid_start_time": _get_pid_start_time(proc.pid),
        "status": "running",
        "command": " ".join(cmd),
        "started_at": time.time(),
    }
    with open(TASK_DIR / f"{task_id}.meta", "w") as f:
        json.dump(meta, f, indent=2)

    return task_id


# ===================================================================
# Centralized task state resolution
# ===================================================================

def _resolve_task_state(task_id: str) -> Dict[str, Any]:
    """
    Single source of truth for task lifecycle state.

    Returns dict with:
      - status: "running" | "completed" | "failed" | "not_found" | "error"
      - task_id
      - meta (if status is running/completed/failed)
      - elapsed_seconds (if status is running/completed/failed)
      - exit_code (if failed, may be None)
      - error (if error/not_found)

    On first detection of process death, persists final state to metadata
    and cleans up _ASYNC_PIDS.
    """
    if not _validate_task_id(task_id):
        return {"status": "error", "task_id": task_id, "error": f"Invalid task ID: {task_id}"}

    meta_file = TASK_DIR / f"{task_id}.meta"
    if not meta_file.exists():
        return {"status": "not_found", "task_id": task_id, "error": f"Task {task_id} not found"}

    try:
        meta = json.loads(meta_file.read_text())
    except Exception as exc:
        return {"status": "error", "task_id": task_id, "error": f"Bad metadata: {exc}"}

    started_at = meta.get("started_at", time.time())

    # Already finalized in a previous call
    if meta.get("status") in ("completed", "failed"):
        completed_at = meta.get("completed_at", started_at)
        return {
            "status": meta["status"],
            "task_id": task_id,
            "meta": meta,
            "elapsed_seconds": int(completed_at - started_at),
            "exit_code": meta.get("exit_code"),
        }

    pid = meta.get("pid")
    pid_start_time = meta.get("pid_start_time")

    if _is_alive(pid, expected_start_time=pid_start_time):
        return {
            "status": "running",
            "task_id": task_id,
            "meta": meta,
            "elapsed_seconds": int(time.time() - started_at),
        }

    # --- Process is dead: finalize under lock ---
    # Lock prevents concurrent threads from both finalizing the same task,
    # which could cause one to lose the exit code after the other pops it.
    lock = _get_task_lock(task_id)
    with lock:
        # Re-read metadata -- another thread may have finalized while we waited
        try:
            meta = json.loads(meta_file.read_text())
        except Exception as exc:
            return {"status": "error", "task_id": task_id, "error": f"Bad metadata: {exc}"}
        if meta.get("status") in ("completed", "failed"):
            completed_at = meta.get("completed_at", started_at)
            return {
                "status": meta["status"],
                "task_id": task_id,
                "meta": meta,
                "elapsed_seconds": int(completed_at - started_at),
                "exit_code": meta.get("exit_code"),
            }

        # Determine exit code.
        # Priority: SIGCHLD handler (reaped with real status) > Popen.poll() > manual waitpid.
        # IMPORTANT: Do NOT call proc.poll() if SIGCHLD already reaped -- Python's
        # waitpid gets ECHILD and silently sets returncode=0, masking real failures.
        exit_code = None
        sigchld_code = _ASYNC_PIDS.get(pid, _NOT_REAPED)
        if sigchld_code is not _NOT_REAPED and sigchld_code is not None:
            # SIGCHLD handler got the real exit code (int, possibly negative for signals)
            exit_code = sigchld_code
        elif sigchld_code is _NOT_REAPED:
            # Handler hasn't reaped yet; Popen.poll() should be safe here
            proc = _ASYNC_PROCS.get(pid)
            if proc is not None:
                proc.poll()
                # Re-check: SIGCHLD handler may have raced between our initial
                # read and proc.poll(), reaping the child first. If so, poll()
                # got ECHILD and set returncode=0 (bogus). Prefer the handler's
                # real exit code.
                raced_code = _ASYNC_PIDS.get(pid, _NOT_REAPED)
                if raced_code is not _NOT_REAPED and raced_code is not None:
                    exit_code = raced_code
                elif proc.returncode is not None:
                    exit_code = proc.returncode
            if exit_code is None:
                # Last resort: manual waitpid
                try:
                    rpid, wstatus = os.waitpid(pid, os.WNOHANG)
                    if rpid == pid:
                        if os.WIFEXITED(wstatus):
                            exit_code = os.WEXITSTATUS(wstatus)
                        elif os.WIFSIGNALED(wstatus):
                            exit_code = -os.WTERMSIG(wstatus)
                        else:
                            exit_code = 127
                except ChildProcessError:
                    pass  # already reaped, exit_code stays None
                except Exception:
                    pass
        # else: sigchld_code is None -- ChildProcessError in handler, exit code lost

        # Use current time as completion timestamp. File mtimes are unreliable --
        # a task that writes output early then runs silently would get a stale
        # timestamp, causing premature cleanup.
        completed_at = time.time()

        # exit_code None means we lost it (race) -- check stderr for error clues
        if exit_code is None:
            stderr_text = _safe_read(TASK_DIR / f"{task_id}.stderr").strip()
            if stderr_text:
                # Non-empty stderr with unknown exit code -- assume failure
                final_status = "failed"
            else:
                final_status = "completed"
        elif exit_code != 0:
            final_status = "failed"
        else:
            final_status = "completed"

        # Persist final state to metadata
        meta["status"] = final_status
        meta["completed_at"] = completed_at
        if exit_code is not None:
            meta["exit_code"] = exit_code
        try:
            meta_file.write_text(json.dumps(meta, indent=2))
        except Exception:
            pass

        # Cleanup in-memory tracking
        _ASYNC_PIDS.pop(pid, None)
        _ASYNC_PROCS.pop(pid, None)

    return {
        "status": final_status,
        "task_id": task_id,
        "meta": meta,
        "elapsed_seconds": int(completed_at - started_at),
        "exit_code": exit_code,
    }


def _check_task(task_id: str) -> Dict[str, Any]:
    """Check task status and return formatted result dict."""
    state = _resolve_task_state(task_id)
    status = state["status"]

    if status in ("error", "not_found"):
        return state

    if status == "running":
        resp: Dict[str, Any] = {
            "status": "running",
            "task_id": task_id,
            "elapsed_seconds": state["elapsed_seconds"],
        }
        # Try to extract thread_id from partial stdout (it's the first line)
        stdout = _safe_read(TASK_DIR / f"{task_id}.stdout")
        if stdout:
            _, tid = _extract_from_jsonl(stdout)
            if tid:
                resp["thread_id"] = tid
        return resp

    # completed or failed
    stdout = _safe_read(TASK_DIR / f"{task_id}.stdout")
    stderr = _safe_read(TASK_DIR / f"{task_id}.stderr")
    result, thread_id = _extract_result(stdout, stderr)

    resp = {
        "status": status,
        "task_id": task_id,
        "result": result,
        "elapsed_seconds": state["elapsed_seconds"],
    }
    if thread_id:
        resp["thread_id"] = thread_id
    if status == "failed":
        resp["exit_code"] = state.get("exit_code")
        if stderr and stderr.strip():
            resp["stderr"] = stderr.strip()[-500:]
    return resp


def _wait_tasks(
    task_ids: List[str],
    timeout: Optional[int] = None,
    request_id: Any = None,
) -> Dict[str, Any]:
    """Block until all tasks complete, timeout, or request is cancelled."""
    deadline = time.time() + timeout if timeout is not None else None
    results = {}

    while True:
        # Check if the MCP client cancelled this request
        if request_id is not None and _is_cancelled(request_id):
            for tid in task_ids:
                if tid not in results:
                    results[tid] = {
                        "status": "cancelled",
                        "task_id": tid,
                        "error": "Wait cancelled by client",
                    }
            break

        pending = []
        for tid in task_ids:
            if tid in results:
                continue
            info = _check_task(tid)
            if info["status"] in ("completed", "failed", "not_found", "error"):
                results[tid] = info
            else:
                pending.append(tid)

        if not pending:
            break

        if deadline is not None and time.time() >= deadline:
            for tid in pending:
                results[tid] = {
                    "status": "timeout",
                    "task_id": tid,
                    "error": "Still running (wait timed out, task NOT killed)",
                }
            break

        sleep_time = 2.0
        if deadline is not None:
            sleep_time = min(sleep_time, max(0.1, deadline - time.time()))
        time.sleep(sleep_time)

    return results

# ===================================================================
# Tool definitions
# ===================================================================

_CODEX_PROPERTIES = {
    "prompt": {
        "type": "string",
        "description": "The initial user prompt for the Codex session.",
    },
    "approval-policy": {
        "type": "string",
        "enum": ["untrusted", "on-failure", "on-request", "never"],
        "description": "Approval policy for shell commands generated by the model.",
    },
    "sandbox": {
        "type": "string",
        "enum": ["read-only", "workspace-write", "danger-full-access"],
        "description": "Sandbox mode.",
    },
    "cwd": {
        "type": "string",
        "description": (
            "Working directory for the session. "
            "If relative, resolved against the server's cwd."
        ),
    },
    "model": {
        "type": "string",
        "description": "Optional override for the model name (e.g. 'gpt-5.4').",
    },
    "profile": {
        "type": "string",
        "description": "Configuration profile from config.toml.",
    },
    "config": {
        "type": "object",
        "additionalProperties": True,
        "description": "Config settings that override server defaults.",
    },
    "base-instructions": {
        "type": "string",
        "description": "Instructions to use instead of the defaults.",
    },
    "developer-instructions": {
        "type": "string",
        "description": "Developer instructions injected as developer role message.",
    },
    "compact-prompt": {
        "type": "string",
        "description": "Prompt used when compacting the conversation.",
    },
}

TOOLS = [
    {
        "name": "codex",
        "description": (
            "Run a Codex session synchronously. "
            "Parameters match the official Codex MCP tool."
        ),
        "inputSchema": {
            "type": "object",
            "properties": _CODEX_PROPERTIES,
            "required": ["prompt"],
        },
    },
    {
        "name": "codex_async",
        "description": (
            "Start a Codex task in the background and return immediately "
            "with a task_id. Use codex_wait to collect results from one or "
            "more tasks, or codex_status to monitor progress."
        ),
        "inputSchema": {
            "type": "object",
            "properties": _CODEX_PROPERTIES,
            "required": ["prompt"],
        },
    },
    {
        "name": "codex_reply",
        "description": (
            "Continue a Codex conversation by providing the thread/session ID "
            "and a follow-up prompt. Uses `codex exec resume` under the hood."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "threadId": {
                    "type": "string",
                    "description": "The session/thread ID (UUID) from a previous Codex call.",
                },
                "prompt": {
                    "type": "string",
                    "description": "The follow-up prompt to continue the conversation.",
                },
            },
            "required": ["prompt", "threadId"],
        },
    },
    {
        "name": "codex_status",
        "description": (
            "Get live status of running async Codex tasks. Shows what each "
            "task is currently doing: last tool call, reasoning, progress. "
            "Works on both running and completed tasks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of task_ids to check status for.",
                },
            },
            "required": ["task_ids"],
        },
    },
    {
        "name": "codex_wait",
        "description": (
            "Block until one or more async Codex tasks complete, then return "
            "all results. Accepts a list of task_ids. This avoids repeated "
            "polling -- call once after launching codex_async tasks. "
            "Default timeout is 1800s (30 min). If a task times out, it is "
            "NOT killed -- it keeps running. You can call codex_wait again "
            "with the same task_ids to resume waiting, or use codex_status "
            "to check progress."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of task_ids to wait for.",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Max seconds to wait (default: 1800). "
                        "The task keeps running even if this times out."
                    ),
                },
            },
            "required": ["task_ids"],
        },
    },
]

# ===================================================================
# Request handler
# ===================================================================

def _handle(request: Dict[str, Any]) -> None:
    method = request.get("method")
    rid = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        _send({
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "codex-mcp-swarm",
                    "version": __version__,
                },
            },
        })
        return

    if method == "notifications/initialized":
        return

    if method == "notifications/cancelled":
        cancelled_id = params.get("requestId")
        if cancelled_id is not None:
            logging.info("Client cancelled request id=%s", cancelled_id)
            with _cancelled_lock:
                _cancelled_requests.add(cancelled_id)
        return

    if method == "tools/list":
        _send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
        return

    if method == "tools/call":
        tool = params.get("name")
        args = params.get("arguments", {})

        if tool == "codex":
            timeout = args.pop("timeout", None)
            result, thread_id = _run_sync(args, timeout=timeout, request_id=rid)
            text = result
            if thread_id:
                text += f"\n\n---\nThread ID: {thread_id}\nUse codex_reply(threadId=\"{thread_id}\", prompt=\"...\") to continue this session."
            _send({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {"content": [{"type": "text", "text": text}]},
            })

        elif tool == "codex_async":
            task_id = _start_async(args)
            _send({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": (
                            f"Codex task started in background.\n"
                            f"Task ID: {task_id}\n\n"
                            f'Use codex_wait(task_ids=["{task_id}"]) to wait '
                            f"for the result, or codex_status(task_ids="
                            f'["{task_id}"]) to check progress.'
                        ),
                    }],
                },
            })

        elif tool == "codex_reply":
            thread_id = args.get("threadId")
            prompt = args.get("prompt")
            timeout = args.get("timeout")

            if not thread_id or not prompt:
                _send({
                    "jsonrpc": "2.0",
                    "id": rid,
                    "error": {
                        "code": -32602,
                        "message": "threadId and prompt are required",
                    },
                })
                return

            text, _ = _run_reply_sync(
                thread_id, prompt, timeout=timeout, request_id=rid
            )
            _send({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {"content": [{"type": "text", "text": text}]},
            })

        elif tool == "codex_status":
            task_ids = args.get("task_ids", [])
            if not task_ids:
                _send({
                    "jsonrpc": "2.0",
                    "id": rid,
                    "error": {"code": -32602, "message": "task_ids is required"},
                })
                return

            parts = []
            for tid in task_ids:
                state = _resolve_task_state(tid)
                status = state["status"]

                if status in ("error", "not_found"):
                    parts.append(f"=== Task {tid} === {state.get('error', status).upper()}")
                    continue

                elapsed = state["elapsed_seconds"]
                stdout_path = TASK_DIR / f"{tid}.stdout"
                jsonl_status = _parse_jsonl_status(stdout_path)

                if status == "running":
                    lines = [f"=== Task {tid} ({elapsed}s elapsed) ==="]
                elif status == "failed":
                    exit_code = state.get("exit_code", "?")
                    lines = [f"=== Task {tid} (FAILED in {elapsed}s, exit {exit_code}) ==="]
                else:
                    lines = [f"=== Task {tid} (COMPLETED in {elapsed}s) ==="]

                # Surface thread_id from meta or stdout
                meta = state.get("meta", {})
                tid_thread = meta.get("thread_id")
                if not tid_thread:
                    stdout_text = _safe_read(stdout_path)
                    if stdout_text:
                        _, tid_thread = _extract_from_jsonl(stdout_text)
                if tid_thread:
                    lines.append(f"Thread ID: {tid_thread}")

                lines.append(f"Phase: {jsonl_status['phase']}")
                lines.append(f"Tools called: {jsonl_status['tools_called']}")

                if jsonl_status["last_tool"]:
                    tool_info = jsonl_status["last_tool"]
                    if jsonl_status["last_tool_args"]:
                        tool_info += f"({jsonl_status['last_tool_args'][:120]})"
                    lines.append(f"Last tool: {tool_info}")

                if jsonl_status["last_reasoning"]:
                    lines.append(
                        f"Thinking: {jsonl_status['last_reasoning'][:200]}"
                    )

                if jsonl_status["last_assistant_text"]:
                    lines.append(
                        f"Output: {jsonl_status['last_assistant_text'][:300]}"
                    )

                parts.append("\n".join(lines))

            _send({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "content": [{"type": "text", "text": "\n\n".join(parts)}],
                },
            })

        elif tool == "codex_wait":
            task_ids = args.get("task_ids", [])
            timeout = args.get("timeout", 1800)

            if not task_ids:
                _send({
                    "jsonrpc": "2.0",
                    "id": rid,
                    "error": {"code": -32602, "message": "task_ids is required"},
                })
                return

            results = _wait_tasks(task_ids, timeout=timeout, request_id=rid)

            parts = []
            for tid in task_ids:
                info = results.get(tid, {"status": "unknown"})
                if info["status"] == "completed":
                    header = (
                        f"=== Task {tid} (completed in "
                        f"{info['elapsed_seconds']}s) ==="
                    )
                    thread_id = info.get("thread_id")
                    if thread_id:
                        header += f"\nThread ID: {thread_id}"
                    parts.append(f"{header}\n{info['result']}")
                elif info["status"] == "failed":
                    exit_code = info.get("exit_code", "?")
                    header = (
                        f"=== Task {tid} (FAILED in "
                        f"{info['elapsed_seconds']}s, exit {exit_code}) ==="
                    )
                    detail = info.get("stderr") or info.get("result", "No output")
                    parts.append(f"{header}\n{detail}")
                elif info["status"] == "timeout":
                    parts.append(
                        f"=== Task {tid} === STILL RUNNING (wait timed out, "
                        f"task is NOT killed -- call codex_wait again to "
                        f"resume waiting)"
                    )
                else:
                    parts.append(
                        f"=== Task {tid} === "
                        f"{info.get('error', info['status'])}"
                    )

            _send({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "content": [{"type": "text", "text": "\n\n".join(parts)}],
                },
            })

        else:
            _send({
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": -32601, "message": f"Unknown tool: {tool}"},
            })
        return

    _send({
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    })

# ===================================================================
# Main
# ===================================================================

def _parse_args() -> None:
    global SERVER_CONFIG, SERVER_FLAGS

    parser = argparse.ArgumentParser(
        description="codex-mcp-swarm -- Parallel Codex MCP Server",
        usage="codex-mcp-swarm [-c key=value]... [--skip-git-repo-check] [--ephemeral]",
    )
    parser.add_argument(
        "-c", "--config",
        action="append",
        default=[],
        metavar="key=value",
        help="Config default in key=value format (repeatable, same as codex mcp-server)",
    )
    parser.add_argument(
        "--skip-git-repo-check",
        action="store_true",
        help="Pass --skip-git-repo-check to all codex exec calls",
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Pass --ephemeral to all codex exec calls (no session persistence)",
    )
    args = parser.parse_args()

    for item in args.config:
        if "=" in item:
            key, value = item.split("=", 1)
            SERVER_CONFIG[key] = value

    if args.skip_git_repo_check:
        SERVER_FLAGS.append("--skip-git-repo-check")
    if args.ephemeral:
        SERVER_FLAGS.append("--ephemeral")


# Methods that can block and must be dispatched to worker threads
_BLOCKING_METHODS = {"tools/call"}


def _handle_threaded(request: Dict[str, Any]) -> None:
    """Wrapper for _handle that catches exceptions in worker threads."""
    rid = request.get("id")
    try:
        _handle(request)
    except Exception as exc:
        logging.error(
            "Handler error (thread): %s\n%s", exc, traceback.format_exc()
        )
        _send({
            "jsonrpc": "2.0",
            "id": rid,
            "error": {
                "code": -32603,
                "message": f"Internal error: {exc}",
            },
        })


def main() -> None:
    _parse_args()
    logging.info(
        "Server starting -- defaults=%s flags=%s", SERVER_CONFIG, SERVER_FLAGS
    )

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    _send({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32600,
                            "message": "Invalid request: expected JSON object",
                        },
                    })
                    continue
                method = request.get("method", "")
                logging.debug("Request: method=%s id=%s", method, request.get("id"))

                if method in _BLOCKING_METHODS:
                    # Dispatch potentially blocking calls to a daemon thread
                    # so the main stdin loop stays responsive
                    t = threading.Thread(
                        target=_handle_threaded,
                        args=(request,),
                        daemon=True,
                    )
                    t.start()
                else:
                    _handle(request)
            except json.JSONDecodeError as exc:
                _send({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                })
            except Exception as exc:
                rid = request.get("id") if isinstance(request, dict) else None
                logging.error(
                    "Handler error: %s\n%s", exc, traceback.format_exc()
                )
                _send({
                    "jsonrpc": "2.0",
                    "id": rid,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {exc}",
                    },
                })
    except KeyboardInterrupt:
        pass

    logging.info("Server stopped")


if __name__ == "__main__":
    main()

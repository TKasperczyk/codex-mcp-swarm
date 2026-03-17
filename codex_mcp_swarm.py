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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__version__ = "1.1.0"

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

# Track async child PIDs so the SIGCHLD handler only reaps those
_ASYNC_PIDS: Dict[int, int] = {}  # pid -> exit_status (set on reap)

# ---------------------------------------------------------------------------
# SIGCHLD handler -- reap only tracked async children
# ---------------------------------------------------------------------------
def _sigchld_handler(signum, frame):
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
            if pid in _ASYNC_PIDS:
                exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
                _ASYNC_PIDS[pid] = exit_code
                logging.debug("Reaped async child PID %d (exit %d)", pid, exit_code)
        except ChildProcessError:
            break
        except Exception as exc:
            logging.warning("SIGCHLD handler error: %s", exc)
            break


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


def _send(response: Dict[str, Any]) -> None:
    try:
        out = json.dumps(response, ensure_ascii=False)
        print(out, flush=True)
        logging.debug("Sent id=%s (%d bytes)", response.get("id"), len(out))
    except (TypeError, ValueError) as exc:
        logging.error("Serialization failed: %s", exc)
        err = {
            "jsonrpc": "2.0",
            "id": response.get("id"),
            "error": {"code": -32603, "message": f"Serialization error: {exc}"},
        }
        print(json.dumps(err), flush=True)


_TASK_ID_RE = re.compile(r"^[0-9a-f]{8}$")


def _validate_task_id(task_id: str) -> bool:
    """Reject task IDs that aren't our generated 8-char hex format."""
    return bool(_TASK_ID_RE.match(task_id))


def _extract_result(stdout: str, stderr: str) -> str:
    """Extract result from codex output. Handles both plain text and JSONL."""
    if stdout.strip().startswith("{"):
        extracted = _extract_from_jsonl(stdout)
        if extracted:
            return extracted
    result = stdout.strip()
    if not result and stderr:
        result = stderr.strip()
    return result or "No output from Codex"


def _extract_from_jsonl(jsonl_text: str) -> Optional[str]:
    """Extract the final assistant message from JSONL output."""
    last_assistant_text = None
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("type") != "response_item":
                continue
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
    return last_assistant_text


def _parse_jsonl_status(stdout_path: Path) -> Dict[str, Any]:
    """
    Parse the JSONL stdout file to determine current Codex activity.
    Returns a dict with: phase, last_tool, last_reasoning, progress.
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
        payload = event.get("payload", {})

        if etype == "event_msg" and payload.get("type") == "task_started":
            status["phase"] = "running"

        elif etype == "response_item":
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
        # Serialize to TOML-compatible values (not Python repr)
        if isinstance(v, bool):
            merged[str(k)] = "true" if v else "false"
        elif isinstance(v, (list, dict)):
            merged[str(k)] = json.dumps(v)
        else:
            merged[str(k)] = str(v)

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
# Sync execution
# ===================================================================

def _run_sync(params: dict, timeout: Optional[int] = None) -> str:
    cmd, cwd = _build_command(params)
    logging.info("Sync exec: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return _extract_result(result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return "Error: Codex execution timed out"
    except Exception as exc:
        return f"Error calling Codex: {exc}"

# ===================================================================
# Async execution
# ===================================================================

def _start_async(params: dict) -> str:
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

    # Track PID for SIGCHLD handler (store sentinel -1, replaced on reap)
    _ASYNC_PIDS[proc.pid] = -1

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


def _check_task(task_id: str) -> Dict[str, Any]:
    if not _validate_task_id(task_id):
        return {"status": "error", "error": f"Invalid task ID: {task_id}"}
    meta_file = TASK_DIR / f"{task_id}.meta"
    if not meta_file.exists():
        return {"status": "not_found", "error": f"Task {task_id} not found"}

    try:
        meta = json.loads(meta_file.read_text())
    except Exception as exc:
        return {"status": "error", "error": f"Bad metadata: {exc}"}

    pid = meta.get("pid")
    pid_start_time = meta.get("pid_start_time")

    if _is_alive(pid, expected_start_time=pid_start_time):
        elapsed = int(time.time() - meta["started_at"])
        return {
            "status": "running",
            "task_id": task_id,
            "elapsed_seconds": elapsed,
        }

    stdout = _safe_read(TASK_DIR / f"{task_id}.stdout")
    stderr = _safe_read(TASK_DIR / f"{task_id}.stderr")
    result = _extract_result(stdout, stderr)

    completed_at = max(
        (TASK_DIR / f"{task_id}.stdout").stat().st_mtime
        if (TASK_DIR / f"{task_id}.stdout").exists() else 0,
        (TASK_DIR / f"{task_id}.stderr").stat().st_mtime
        if (TASK_DIR / f"{task_id}.stderr").exists() else 0,
    ) or time.time()

    meta["status"] = "completed"
    meta["completed_at"] = completed_at
    try:
        meta_file.write_text(json.dumps(meta, indent=2))
    except Exception:
        pass

    return {
        "status": "completed",
        "task_id": task_id,
        "result": result,
        "elapsed_seconds": int(completed_at - meta["started_at"]),
    }


def _wait_tasks(task_ids: List[str], timeout: Optional[int] = None) -> Dict[str, Any]:
    """Block until all tasks complete (or timeout)."""
    deadline = time.time() + timeout if timeout else None
    results = {}

    while True:
        pending = []
        for tid in task_ids:
            if tid in results:
                continue
            info = _check_task(tid)
            if info["status"] in ("completed", "not_found", "error"):
                results[tid] = info
            else:
                pending.append(tid)

        if not pending:
            break

        if deadline and time.time() >= deadline:
            for tid in pending:
                results[tid] = {
                    "status": "timeout",
                    "task_id": tid,
                    "error": "Still running (wait timed out, task NOT killed)",
                }
            break

        time.sleep(2)

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
            "properties": {
                **_CODEX_PROPERTIES,
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: no limit). Avoid setting this below 1800 for complex tasks.",
                },
            },
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
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: no limit). Avoid setting this below 1800 for complex tasks.",
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

    if method == "tools/list":
        _send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
        return

    if method == "tools/call":
        tool = params.get("name")
        args = params.get("arguments", {})

        if tool == "codex":
            timeout = args.pop("timeout", None)
            result = _run_sync(args, timeout=timeout)
            _send({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {"content": [{"type": "text", "text": result}]},
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

            cmd = _build_reply_command(thread_id, prompt)
            logging.info("Reply exec: %s", " ".join(cmd))
            try:
                result = subprocess.run(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                text = _extract_result(result.stdout, result.stderr)
            except subprocess.TimeoutExpired:
                text = "Error: Codex reply timed out"
            except Exception as exc:
                text = f"Error calling Codex reply: {exc}"

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
                if not _validate_task_id(tid):
                    parts.append(f"=== Task {tid} === INVALID ID")
                    continue
                meta_file = TASK_DIR / f"{tid}.meta"
                if not meta_file.exists():
                    parts.append(f"=== Task {tid} === NOT FOUND")
                    continue

                try:
                    meta = json.loads(meta_file.read_text())
                except Exception:
                    parts.append(f"=== Task {tid} === ERROR reading metadata")
                    continue

                alive = _is_alive(meta.get("pid"))
                elapsed = int(time.time() - meta.get("started_at", time.time()))
                jsonl_status = _parse_jsonl_status(TASK_DIR / f"{tid}.stdout")

                lines = [f"=== Task {tid} ({elapsed}s elapsed) ==="]
                if not alive:
                    lines[0] = f"=== Task {tid} (COMPLETED in {elapsed}s) ==="

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

            results = _wait_tasks(task_ids, timeout=timeout)

            parts = []
            for tid in task_ids:
                info = results.get(tid, {"status": "unknown"})
                if info["status"] == "completed":
                    parts.append(
                        f"=== Task {tid} (completed in "
                        f"{info['elapsed_seconds']}s) ===\n"
                        f"{info['result']}"
                    )
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
            rid = None
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
                rid = request.get("id")
                logging.debug("Request: method=%s id=%s", request.get("method"), rid)
                _handle(request)
            except json.JSONDecodeError as exc:
                _send({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                })
            except Exception as exc:
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

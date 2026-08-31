#!/usr/bin/env python3
"""
codex-mcp-swarm -- Parallel Codex MCP Server

An MCP server that wraps OpenAI's Codex CLI with true parallel execution,
live task monitoring, and full parameter parity with the official Codex
MCP tool.

Features:
  - codex:        Synchronous execution (drop-in replacement)
  - codex_async:  Launch a task, return a task_id immediately (fan-out).
                  NOT fire-and-forget: the id is server-side only, so the
                  result reaches the caller solely via codex_wait.
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
import shlex
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

__version__ = "1.11.0"

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

WORKTREE_BASE_DIR = Path(
    os.environ.get("CODEX_SWARM_WORKTREE_DIR", "/tmp/codex-swarm-worktrees")
)
WORKTREE_BASE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

# ---------------------------------------------------------------------------
# Protocol negotiation
# ---------------------------------------------------------------------------
# Newest first. The wire surface this server actually uses (tools, resources,
# notifications/progress) is identical across every one of these, so picking a
# revision is a formality. It stopped being a formality when the response was
# hardcoded to 2024-11-05: Claude Code 2.1.232 asks for 2025-11-25 and was told
# the server speaks a revision three behind it, which opts out of everything
# added since for no reason at all. Echo what the client asked for when we know
# it, and fall back to our newest otherwise, which is what the spec requires.
_SUPPORTED_PROTOCOL_VERSIONS = (
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
_DEFAULT_PROTOCOL_VERSION = _SUPPORTED_PROTOCOL_VERSIONS[0]

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


def _create_worktree(run_id: str, base_cwd: Optional[str]) -> Tuple[str, str, str]:
    """Create an isolated worktree. Returns (worktree_root, target_cwd, branch_name)."""
    branch_name = f"codex-swarm/{run_id}"
    worktree_root = WORKTREE_BASE_DIR / run_id

    if base_cwd:
        base_path = Path(base_cwd)
        effective_cwd = base_path if base_path.is_absolute() else Path.cwd() / base_path
    else:
        effective_cwd = Path.cwd()
    effective_cwd = effective_cwd.resolve()

    try:
        proc = subprocess.run(
            ["git", "-C", str(effective_cwd), "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        repo_root = Path(proc.stdout.strip()).resolve()
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"Failed to find git repo for {effective_cwd}: {detail}") from exc

    try:
        rel_subdir = effective_cwd.relative_to(repo_root)
    except ValueError:
        rel_subdir = Path(".")

    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "worktree",
                "add",
                "-b",
                branch_name,
                str(worktree_root),
                "HEAD",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(
            f"Failed to create git worktree {worktree_root}: {detail}"
        ) from exc

    target_cwd = (worktree_root / rel_subdir).resolve()
    logging.info(
        "Created worktree %s (branch %s) for run %s", worktree_root, branch_name, run_id
    )
    return str(worktree_root), str(target_cwd), branch_name


def _remove_worktree(worktree_path: str, branch_name: str) -> bool:
    """Remove a worktree and its branch. Returns True if cleanup succeeded."""
    worktree_input = Path(worktree_path)
    if not worktree_input.exists():
        logging.debug("Worktree path already removed: %s", worktree_input)
        return True

    worktree_root = worktree_input
    repo_root: Optional[Path] = None

    try:
        proc = subprocess.run(
            ["git", "-C", str(worktree_input), "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        worktree_root = Path(proc.stdout.strip()).resolve()
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        logging.debug("Unable to resolve worktree root for %s: %s", worktree_input, detail)

    try:
        proc = subprocess.run(
            ["git", "-C", str(worktree_input), "rev-parse", "--git-common-dir"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        common_dir = Path(proc.stdout.strip())
        if not common_dir.is_absolute():
            common_dir = (worktree_input / common_dir).resolve()
        if common_dir.name == ".git":
            repo_root = common_dir.parent
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        logging.debug("Unable to resolve git common dir for %s: %s", worktree_input, detail)

    if worktree_root.exists():
        remove_cwd = repo_root or worktree_root
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(remove_cwd),
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree_root),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            logging.info("Removed worktree %s", worktree_root)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            lowered = detail.lower()
            if not any(
                token in lowered
                for token in (
                    "not a git repository",
                    "working tree not found",
                    "is not a working tree",
                    "does not exist",
                    "no such file",
                )
            ):
                logging.warning("Failed to remove worktree %s: %s", worktree_root, detail)
                return False

    if branch_name and repo_root:
        try:
            subprocess.run(
                ["git", "-C", str(repo_root), "branch", "-D", branch_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            logging.info("Deleted worktree branch %s", branch_name)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            logging.debug("Ignoring worktree branch delete failure for %s: %s", branch_name, detail)

    return True


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


def _error_message(value: Any) -> Optional[str]:
    """Return a non-empty message from a public exec error payload."""
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, dict):
        message = value.get("message")
        if isinstance(message, str):
            message = message.strip()
            return message or None
    return None


def _analyze_jsonl(jsonl_text: str) -> Dict[str, Any]:
    """Analyze exec/session JSONL without treating diagnostic errors as terminal.

    Codex 0.151 emits retryable stream errors as top-level ``error`` events but
    strips the internal ``will_retry`` flag from public JSONL. Consequently only
    ``turn.completed`` and ``turn.failed`` are terminal. Error items are warnings.
    If both terminal event types occur, the stream is contradictory and the
    caller must prefer an observed process exit code.
    """
    analysis: Dict[str, Any] = {
        "thread_id": None,
        "last_assistant_text": None,
        "errors": [],
        "warnings": [],
        "terminal_events": [],
        "terminal_status": None,
        "terminal_error": None,
        "terminal_conflict": False,
        "saw_json": False,
        "saw_turn_started": False,
    }

    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        analysis["saw_json"] = True
        etype = event.get("type", "")

        if etype == "thread.started":
            analysis["thread_id"] = event.get("thread_id")
        elif etype == "session_meta":
            payload = event.get("payload", {})
            if isinstance(payload, dict) and not analysis["thread_id"]:
                analysis["thread_id"] = payload.get("id")

        if etype == "turn.started":
            analysis["saw_turn_started"] = True
        elif etype == "turn.completed":
            analysis["terminal_events"].append("completed")
        elif etype == "turn.failed":
            analysis["terminal_events"].append("failed")
            message = _error_message(event.get("error"))
            if message:
                analysis["terminal_error"] = message
        elif etype == "error":
            # Non-terminal: this may be a retry notification whose will_retry
            # flag was removed by the public exec JSONL representation.
            message = _error_message(event.get("message")) or _error_message(
                event.get("error")
            )
            if message:
                analysis["errors"].append(message)
        elif etype == "item.completed":
            item = event.get("item", {})
            if isinstance(item, dict):
                itype = item.get("type")
                if itype == "agent_message":
                    text = item.get("text", "")
                    if isinstance(text, str) and text:
                        analysis["last_assistant_text"] = text
                elif itype == "error":
                    # Config notices, runtime warnings and model reroutes use
                    # this form. They are explicitly non-terminal.
                    message = _error_message(item.get("message")) or _error_message(
                        item.get("error")
                    )
                    if message:
                        analysis["warnings"].append(message)
        elif etype == "response_item":
            # Session rollout format (fallback for callers that inspect one).
            payload = event.get("payload", {})
            if (
                isinstance(payload, dict)
                and payload.get("role") == "assistant"
                and payload.get("type") == "message"
            ):
                content = payload.get("content") or []
                texts = [
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict)
                    and c.get("type") in ("output_text", "text", "input_text")
                ]
                if texts:
                    analysis["last_assistant_text"] = "\n".join(texts)
        elif etype == "event_msg":
            # Persisted rollouts use event_msg rather than public exec events.
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                continue
            ptype = payload.get("type")
            if ptype == "task_started":
                analysis["saw_turn_started"] = True
            elif ptype == "task_complete":
                analysis["terminal_events"].append("completed")
                message = payload.get("last_agent_message")
                if isinstance(message, str) and message:
                    analysis["last_assistant_text"] = message
            elif ptype == "turn_aborted":
                analysis["terminal_events"].append("failed")
                reason = payload.get("reason")
                if isinstance(reason, str) and reason:
                    analysis["terminal_error"] = f"Turn aborted: {reason}"
            elif ptype == "error":
                message = _error_message(payload.get("message")) or _error_message(
                    payload.get("error")
                )
                if message:
                    analysis["errors"].append(message)

    terminal_types = set(analysis["terminal_events"])
    if len(terminal_types) == 1:
        analysis["terminal_status"] = next(iter(terminal_types))
    elif len(terminal_types) > 1:
        analysis["terminal_conflict"] = True
    return analysis


def _decide_run_status(
    exit_code: Optional[int], analysis: Dict[str, Any], stderr: str
) -> Tuple[str, bool]:
    """Return (status, inferred) using terminal events, then the exit code."""
    terminal_status = analysis.get("terminal_status")
    if terminal_status in ("completed", "failed"):
        return terminal_status, False

    # Missing or contradictory terminal signals defer to the observed process.
    if exit_code is not None:
        return ("failed" if exit_code != 0 else "completed"), False

    # Exit-code-lost compatibility fallback from 1.10.0. Agent output is useful
    # evidence of completion; stderr is only a weak, explicitly inferred signal.
    if analysis.get("last_assistant_text"):
        return "completed", True
    if stderr.strip():
        return "failed", True
    return "completed", True


def _model_from_command(command: Any) -> Optional[str]:
    """Extract the effective ``-m`` or ``-c model=`` value from task metadata."""
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    for index, token in enumerate(tokens):
        if token in ("-m", "--model") and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith("--model="):
            return token.split("=", 1)[1] or None

    for index, token in enumerate(tokens):
        config_value = None
        if token in ("-c", "--config") and index + 1 < len(tokens):
            config_value = tokens[index + 1]
        elif token.startswith("--config="):
            config_value = token.split("=", 1)[1]
        if config_value and config_value.startswith("model="):
            return config_value.split("=", 1)[1] or None
    return None


def _classify_failure(message: str) -> Dict[str, Any]:
    """Conservatively classify a final failure; retryability is information only."""
    lower = message.lower()
    category = "unknown"
    retryable: Optional[bool] = None
    action = "Inspect the structured cause and diagnostics before retrying."

    if any(term in lower for term in (
        "refresh token", "token has been revoked", "revoked token",
        "unauthorized", "authentication failed", "not authenticated",
    )) or re.search(r"\b401\b", lower):
        category, retryable = "authentication", False
        action = "Reauthenticate Codex before starting another run."
    elif any(term in lower for term in (
        "at capacity", "model capacity", "high demand", "provider capacity",
        "server overloaded", "service overloaded",
    )):
        category, retryable = "provider_capacity", True
        action = "Retry later, or ask the server operator to change the configured model."
    elif any(term in lower for term in (
        "rate limit", "rate-limit", "too many requests", "quota temporarily",
    )) or re.search(r"\b429\b", lower):
        category, retryable = "rate_limit", True
        action = "Retry after the provider limit resets."
    elif any(term in lower for term in (
        "internal server error", "bad gateway", "service unavailable",
        "gateway timeout",
    )) or re.search(r"\b5\d\d\b", lower):
        category, retryable = "provider_5xx", True
        action = "Retry later; the provider or an upstream gateway failed."
    elif any(term in lower for term in (
        "stream disconnected", "connection reset", "connection failed",
        "network error", "timed out", "timeout",
    )):
        category, retryable = "transport", True
        action = "Retry after checking network and provider availability."
    elif any(term in lower for term in (
        "invalid request", "bad request", "invalid prompt", "unsupported",
        "model not found", "unknown model",
    )) or re.search(r"\b400\b", lower):
        category, retryable = "invalid_request", False
        action = "Correct the request or server configuration before retrying."
    elif any(term in lower for term in (
        "panicked at", "panic", "segmentation fault", "fatal runtime error",
        "agent loop died unexpectedly", "signal 11",
    )):
        category = "process_crash"
        action = "Inspect the diagnostics; retryability is unknown."

    return {"category": category, "retryable": retryable, "action": action}


def _build_failure_info(
    stdout: str,
    stderr: str,
    command: Any = None,
    analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a clean primary cause plus explicitly secondary diagnostics."""
    analysis = analysis or _analyze_jsonl(stdout)
    cause = analysis.get("terminal_error")
    if not cause and analysis.get("errors"):
        cause = analysis["errors"][-1]

    raw_stdout = stdout.strip()
    stdout_diagnostics = None
    if not cause and raw_stdout and not analysis.get("saw_json"):
        cause = raw_stdout
    elif not cause and raw_stdout:
        stdout_diagnostics = raw_stdout[-1000:]

    if not cause:
        cause = "Codex exited without a structured failure message."

    classification_input = cause
    if classification_input.startswith("Codex exited without") and stderr.strip():
        classification_input = f"{classification_input}\n{stderr}"
    classification = _classify_failure(classification_input)

    info: Dict[str, Any] = {
        "cause": cause,
        "model": _model_from_command(command),
        **classification,
    }
    if stdout_diagnostics:
        info["stdout_diagnostics"] = stdout_diagnostics
    if stderr.strip():
        info["stderr_diagnostics"] = stderr.strip()[-500:]
    return info


def _format_failure_info(
    info: Dict[str, Any], include_diagnostics: bool = True
) -> str:
    retryable = info.get("retryable")
    retryable_text = "yes" if retryable is True else "no" if retryable is False else "unknown"
    lines = [
        f"Model: {info.get('model') or 'unknown'}",
        f"Cause: {info.get('cause', 'Unknown Codex failure')}",
        f"Category: {info.get('category', 'unknown')}",
        f"Retryable: {retryable_text} (informational only; the wrapper did not retry)",
    ]
    if info.get("action"):
        lines.append(f"Action: {info['action']}")
    if include_diagnostics and info.get("stdout_diagnostics"):
        lines.extend([
            "",
            "Output (stdout; no structured cause was found):",
            info["stdout_diagnostics"],
        ])
    if include_diagnostics and info.get("stderr_diagnostics"):
        lines.extend([
            "",
            "Diagnostics (stderr tail; not the failure cause):",
            info["stderr_diagnostics"],
        ])
    return "\n".join(lines)


def _extract_result(
    stdout: str, stderr: str, analysis: Optional[Dict[str, Any]] = None
) -> Tuple[str, Optional[str]]:
    """Extract result and thread ID from codex output. Returns (text, thread_id)."""
    analysis = analysis or _analyze_jsonl(stdout)
    if stdout.strip().startswith("{") and analysis.get("last_assistant_text"):
        return analysis["last_assistant_text"], analysis.get("thread_id")
    result = stdout.strip()
    if not result and stderr:
        result = stderr.strip()
    return result or "No output from Codex", analysis.get("thread_id")


def _extract_from_jsonl(jsonl_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract the final assistant message and thread ID from analyzed JSONL."""
    analysis = _analyze_jsonl(jsonl_text)
    return analysis.get("last_assistant_text"), analysis.get("thread_id")


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
        "last_error": None,
        "last_warning": None,
        "terminal_error": None,
        "terminal_conflict": False,
    }

    if not stdout_path.exists():
        return status

    try:
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return status

    analysis = _analyze_jsonl(text)

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

    terminal_status = analysis.get("terminal_status")
    if terminal_status:
        status["phase"] = terminal_status
    elif analysis.get("terminal_conflict"):
        status["phase"] = "unknown (conflicting terminal events)"
    status["last_error"] = analysis["errors"][-1] if analysis["errors"] else None
    status["last_warning"] = analysis["warnings"][-1] if analysis["warnings"] else None
    status["terminal_error"] = analysis.get("terminal_error")
    status["terminal_conflict"] = bool(analysis.get("terminal_conflict"))
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

def _with_json_flag(cmd: List[str]) -> List[str]:
    """
    Insert --json after the subcommand.

    Position matters: `codex exec` takes it at index 2, but `codex exec resume`
    needs index 3 or the flag lands where the subcommand belongs.
    """
    if "--json" in cmd:
        return cmd
    idx = 3 if len(cmd) > 2 and cmd[2] == "resume" else 2
    return cmd[:idx] + ["--json"] + cmd[idx:]


def _build_command(params: dict) -> Tuple[List[str], Optional[str]]:
    """
    Build a `codex exec` command from tool parameters + server defaults.

    With params["threadId"] this builds `codex exec resume <id>` instead. The
    resume subcommand rejects -s/-p/-C at arg-parse time (exit 2), so on
    resume the sandbox is passed as a -c sandbox_mode override, the cwd is
    applied via the subprocess working directory only, and a profile cannot
    be expressed at all (`-c profile=` is rejected as legacy config).

    Returns (cmd_list, cwd_or_none).
    """
    cmd = ["codex", "exec"]

    thread_id = params.get("threadId")
    is_resume = bool(thread_id)
    if is_resume:
        cmd.append("resume")

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
        merged["sandbox_mode"] = sandbox
    if not is_resume and "sandbox_mode" in merged:
        cmd.extend(["-s", merged.pop("sandbox_mode")])

    # Approval policy (no dedicated flag -- stays as -c)
    approval = params.get("approval-policy")
    if approval:
        merged["approval_policy"] = approval

    # Profile
    profile = params.get("profile")
    if profile:
        if is_resume:
            logging.warning(
                "Resume of %s: dropping profile=%r "
                "(codex exec resume has no --profile flag)",
                thread_id, profile,
            )
        else:
            cmd.extend(["-p", profile])

    # CWD
    cwd = params.get("cwd")
    if cwd and not is_resume:
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

    if thread_id:
        cmd.append(str(thread_id))

    prompt = params.get("prompt", "")
    if prompt:
        cmd.append(prompt)

    # Fresh exec passes -C, so a relative cwd resolves inside codex against
    # the server's cwd; resume has no -C, so resolve it here instead.
    if is_resume:
        proc_cwd = os.path.abspath(cwd) if cwd else None
    else:
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
_PROGRESS_INTERVAL = 20  # seconds between heartbeat progress notifications

# Claude Code moves a main-conversation tool call that is still running after
# two minutes into a tracked background task, and re-invokes the session with
# the result when it settles. That notification is the ONLY channel that wakes
# an idle session, so codex_wait wants to cross this line, not duck under it.
# A wait that returns at 115s hands the job of staying awake back to the model,
# and a model that forgets loses the result entirely (observed 2026-08-14:
# task 74291012 ran to completion with nobody left to collect it).
_CLIENT_AUTO_BACKGROUND_S = 120

# Floor for codex_wait, comfortably past the backgrounding threshold. Tasks
# that are ALREADY finished still return immediately -- _wait_tasks resolves
# every task before it ever consults the deadline -- so this only extends waits
# that would have come back empty-handed anyway.
_WAIT_MIN_TIMEOUT = int(
    os.environ.get("CODEX_SWARM_MIN_WAIT", _CLIENT_AUTO_BACKGROUND_S + 30)
)

# A wrapper-level "failure" is not proof the Codex run died or that its edits
# never landed. Both of these are attached to output that callers have
# historically acted on as if it were a clean failure.
_UNVERIFIED_FAILURE_HINT = (
    "NOTE: this verdict is inferred, not observed -- the exit code was lost to "
    "a reaping race, so the status was derived from output alone. Codex may "
    "have finished its work. Verify before reporting failure upstream: run "
    "`pgrep -af 'codex exec'` to see whether it is still alive, then check "
    "file mtimes, the worktree branch, and your own build/tests."
)

_UNVERIFIED_COMPLETION_HINT = (
    "NOTE: this completion verdict is inferred, not observed -- neither an "
    "unambiguous terminal event nor the process exit code was available. "
    "Verify the returned output and your own build/tests before relying on it."
)

_STILL_RUNNING_HINT = (
    "This is NOT a failure. The task was not killed and is still working. Call "
    "codex_wait again with the same task_id to keep waiting, or check "
    "`pgrep -af 'codex exec'`."
)

# Sync children, so an abandoned or cancelled request does not leave an
# untracked `codex exec` running with no handle anywhere in the server.
_SYNC_PROCS: Dict[Any, subprocess.Popen] = {}
_sync_procs_lock = threading.Lock()


def _register_sync_proc(request_id: Any, proc: subprocess.Popen) -> None:
    if request_id is None:
        return
    with _sync_procs_lock:
        _SYNC_PROCS[request_id] = proc


def _unregister_sync_proc(request_id: Any) -> None:
    if request_id is None:
        return
    with _sync_procs_lock:
        _SYNC_PROCS.pop(request_id, None)


def _terminate_sync_proc(request_id: Any) -> bool:
    """Terminate the sync child for a request, if one is still running."""
    with _sync_procs_lock:
        proc = _SYNC_PROCS.get(request_id)
    if proc is None or proc.poll() is not None:
        return False
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    logging.info("Terminated sync child pid=%s for request id=%s", proc.pid, request_id)
    return True


def _terminate_all_sync_procs() -> None:
    """Kill any surviving sync children on server shutdown."""
    with _sync_procs_lock:
        items = list(_SYNC_PROCS.items())
    for request_id, proc in items:
        if proc.poll() is None:
            logging.info("Shutdown: killing sync child pid=%s (request %s)", proc.pid, request_id)
            proc.kill()


def _send_progress(progress_token: Any, elapsed: float, label: str) -> None:
    """
    Emit an MCP progress notification.

    This is what keeps a long sync run alive: the client's idle timer measures
    silence, not duration, so a run with no wire traffic is indistinguishable
    from a hung server no matter how much real work it is doing.
    """
    if progress_token is None:
        return
    _send({
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {
            "progressToken": progress_token,
            "progress": round(elapsed, 1),
            "message": f"{label} running ({int(elapsed)}s elapsed)",
        },
    })


def _wait_proc(
    proc: subprocess.Popen,
    deadline: Optional[float] = None,
    request_id: Any = None,
    timeout_msg: str = "Error: Codex execution timed out",
    progress_token: Any = None,
    progress_label: str = "Codex",
    command: Any = None,
) -> Tuple[str, Optional[str], bool]:
    """
    Wait for a Popen process with cancellation, timeout and progress support.
    Shared by sync codex and reply paths.
    """
    started = time.time()
    last_progress = started
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
                return timeout_msg, None, True
        wait_time = min(_POLL_INTERVAL, remaining) if remaining is not None else _POLL_INTERVAL
        try:
            stdout, stderr = proc.communicate(timeout=wait_time)
            analysis = _analyze_jsonl(stdout)
            status, _ = _decide_run_status(proc.returncode, analysis, stderr)
            if status == "failed":
                failure = _build_failure_info(
                    stdout, stderr, command=command, analysis=analysis
                )
                exit_code = proc.returncode if proc.returncode is not None else "?"
                text = (
                    f"{progress_label} FAILED (exit {exit_code})\n"
                    f"{_format_failure_info(failure)}"
                )
                return text, analysis.get("thread_id"), True
            result, thread_id = _extract_result(stdout, stderr, analysis=analysis)
            return result, thread_id, False
        except subprocess.TimeoutExpired:
            if request_id is not None and _is_cancelled(request_id):
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return "Cancelled by client", None, True
            now = time.time()
            if now - last_progress >= _PROGRESS_INTERVAL:
                last_progress = now
                _send_progress(progress_token, now - started, progress_label)


def _run_sync(
    params: dict,
    request_id: Any = None,
    progress_token: Any = None,
) -> Tuple[str, Optional[str], Optional[Dict[str, str]]]:
    """
    Run codex synchronously with cancellation support.

    No timeout by design -- see _run_reply_sync for the reasoning.
    """
    started_at = time.time()
    cmd_params = dict(params)
    worktree_info: Optional[Dict[str, str]] = None
    run_id: Optional[str] = None
    cmd: List[str] = []
    thread_id: Optional[str] = None
    exit_code: Optional[int] = None
    meta_status = "failed"

    worktree_enabled = bool(cmd_params.pop("worktree", False))

    try:
        if worktree_enabled:
            _cleanup_old_tasks()
            run_id = uuid.uuid4().hex[:8]
            wt_root, wt_cwd, worktree_branch = _create_worktree(run_id, cmd_params.get("cwd"))
            cmd_params["cwd"] = wt_cwd
            worktree_info = {
                "worktree_root": wt_root,
                "worktree_path": wt_cwd,
                "worktree_branch": worktree_branch,
            }

        cmd, cwd = _build_command(cmd_params)
        # Add --json for structured output (enables thread_id extraction)
        cmd = _with_json_flag(cmd)
        logging.info("Sync exec: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )
        _register_sync_proc(request_id, proc)
        result, thread_id, failed = _wait_proc(
            proc,
            request_id=request_id,
            timeout_msg="Error: Codex execution timed out",
            progress_token=progress_token,
            progress_label="Codex",
            command=" ".join(cmd),
        )
        exit_code = proc.returncode
        meta_status = "failed" if failed else "completed"
        return result, thread_id, worktree_info
    except Exception as exc:
        return f"Error calling Codex: {exc}", None, worktree_info
    finally:
        _unregister_sync_proc(request_id)
        if worktree_info and run_id:
            meta = {
                "task_id": run_id,
                "status": meta_status,
                "started_at": started_at,
                "completed_at": time.time(),
                "worktree_root": worktree_info["worktree_root"],
                "worktree_path": worktree_info["worktree_path"],
                "worktree_branch": worktree_info["worktree_branch"],
            }
            if cmd:
                meta["command"] = " ".join(cmd)
            if exit_code is not None:
                meta["exit_code"] = exit_code
            if thread_id:
                meta["thread_id"] = thread_id
            try:
                with open(TASK_DIR / f"{run_id}.meta", "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception as exc:
                logging.warning("Failed to persist sync worktree metadata for %s: %s", run_id, exc)


def _run_reply_sync(
    thread_id: str,
    prompt: str,
    request_id: Any = None,
    progress_token: Any = None,
) -> Tuple[str, Optional[str]]:
    """
    Run codex reply synchronously with cancellation support.

    Deliberately has no timeout: sync tools run to completion, and callers that
    need a bounded wait should use codex_async(threadId=...) + codex_wait.
    Models set a low timeout here whatever the description says, which turned
    long resumes into spurious failures.
    """
    cmd = _with_json_flag(_build_reply_command(thread_id, prompt))
    logging.info("Reply exec: %s", " ".join(cmd))
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _register_sync_proc(request_id, proc)
        result, result_thread_id, _ = _wait_proc(
            proc,
            request_id=request_id,
            timeout_msg="Error: Codex reply timed out",
            progress_token=progress_token,
            progress_label="Codex reply",
            command=" ".join(cmd),
        )
        return result, result_thread_id
    except Exception as exc:
        return f"Error calling Codex reply: {exc}", None
    finally:
        if proc is not None:
            _unregister_sync_proc(request_id)

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

            if status not in ("completed", "failed", "cancelled"):
                continue
            completed_at = meta.get("completed_at", 0)
            if now - completed_at < _TASK_MAX_AGE:
                continue

            # Clean up worktree if present -- use worktree_root (stable path)
            wt_root = meta.get("worktree_root") or meta.get("worktree_path")
            worktree_branch = meta.get("worktree_branch")
            if wt_root:
                if not _remove_worktree(wt_root, worktree_branch or ""):
                    # Worktree removal failed -- keep metadata for retry next sweep
                    logging.warning("Deferring cleanup of task %s (worktree removal failed)", task_id)
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


def _start_async(params: dict) -> Dict[str, Optional[str]]:
    _cleanup_old_tasks()

    task_id = uuid.uuid4().hex[:8]
    cmd_params = dict(params)
    worktree_enabled = bool(cmd_params.pop("worktree", False))
    worktree_root: Optional[str] = None
    worktree_path: Optional[str] = None
    worktree_branch: Optional[str] = None

    if worktree_enabled:
        worktree_root, wt_cwd, worktree_branch = _create_worktree(task_id, cmd_params.get("cwd"))
        worktree_path = wt_cwd
        cmd_params["cwd"] = wt_cwd

    try:
        cmd, cwd = _build_command(cmd_params)
        # Add --json for structured output (enables live status parsing)
        cmd = _with_json_flag(cmd)
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
    except Exception:
        # Clean up worktree if task launch fails after creation
        if worktree_root and worktree_branch:
            _remove_worktree(worktree_root, worktree_branch)
        raise

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
    if worktree_root:
        meta["worktree_root"] = worktree_root
    if worktree_path:
        meta["worktree_path"] = worktree_path
    if worktree_branch:
        meta["worktree_branch"] = worktree_branch
    with open(TASK_DIR / f"{task_id}.meta", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "task_id": task_id,
        "worktree_path": worktree_path,
        "worktree_branch": worktree_branch,
    }


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
      - exit_code_lost (whether the process exit code was unavailable)
      - status_inferred (whether the verdict used the compatibility fallback)
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
    if meta.get("status") in ("completed", "failed", "cancelled"):
        completed_at = meta.get("completed_at", started_at)
        status_inferred = (
            bool(meta["status_inferred"])
            if "status_inferred" in meta
            else bool(meta.get("exit_code_lost"))
        )
        return {
            "status": meta["status"],
            "task_id": task_id,
            "meta": meta,
            "elapsed_seconds": int(completed_at - started_at),
            "exit_code": meta.get("exit_code"),
            "exit_code_lost": bool(meta.get("exit_code_lost")),
            "status_inferred": status_inferred,
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
        if meta.get("status") in ("completed", "failed", "cancelled"):
            completed_at = meta.get("completed_at", started_at)
            status_inferred = (
                bool(meta["status_inferred"])
                if "status_inferred" in meta
                else bool(meta.get("exit_code_lost"))
            )
            return {
                "status": meta["status"],
                "task_id": task_id,
                "meta": meta,
                "elapsed_seconds": int(completed_at - started_at),
                "exit_code": meta.get("exit_code"),
                "exit_code_lost": bool(meta.get("exit_code_lost")),
                "status_inferred": status_inferred,
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

        # Unambiguous terminal JSONL is authoritative even if the exit code was
        # lost: turn.completed wins over earlier retryable errors, while
        # turn.failed wins over an earlier partial agent message. Missing or
        # contradictory terminal events defer to the observed exit code. Only
        # when that code was also lost do we use the 1.10.0 compatibility
        # fallback, whose stderr branch is explicitly marked as inferred.
        exit_code_lost = exit_code is None
        stdout_text = _safe_read(TASK_DIR / f"{task_id}.stdout")
        stderr_text = _safe_read(TASK_DIR / f"{task_id}.stderr")
        analysis = _analyze_jsonl(stdout_text)
        final_status, status_inferred = _decide_run_status(
            exit_code, analysis, stderr_text
        )

        # Persist final state to metadata
        meta["status"] = final_status
        meta["completed_at"] = completed_at
        if exit_code is not None:
            meta["exit_code"] = exit_code
        if exit_code_lost:
            meta["exit_code_lost"] = True
        meta["status_inferred"] = status_inferred
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
        "exit_code_lost": exit_code_lost,
        "status_inferred": status_inferred,
    }


def _check_task(task_id: str) -> Dict[str, Any]:
    """Check task status and return formatted result dict."""
    state = _resolve_task_state(task_id)
    status = state["status"]

    if status in ("error", "not_found"):
        return state

    if status == "running":
        meta = state.get("meta", {})
        resp: Dict[str, Any] = {
            "status": "running",
            "task_id": task_id,
            "elapsed_seconds": state["elapsed_seconds"],
        }
        if meta.get("worktree_path"):
            resp["worktree_path"] = meta["worktree_path"]
        if meta.get("worktree_branch"):
            resp["worktree_branch"] = meta["worktree_branch"]
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
    analysis = _analyze_jsonl(stdout)
    result, thread_id = _extract_result(stdout, stderr, analysis=analysis)
    meta = state.get("meta", {})

    resp = {
        "status": status,
        "task_id": task_id,
        "result": result,
        "elapsed_seconds": state["elapsed_seconds"],
    }
    if status in ("completed", "failed"):
        resp["exit_code"] = state.get("exit_code")
        resp["exit_code_lost"] = bool(state.get("exit_code_lost"))
        resp["status_inferred"] = bool(state.get("status_inferred"))
    if meta.get("worktree_path"):
        resp["worktree_path"] = meta["worktree_path"]
    if meta.get("worktree_branch"):
        resp["worktree_branch"] = meta["worktree_branch"]
    if thread_id:
        resp["thread_id"] = thread_id
    if status == "failed":
        failure = _build_failure_info(
            stdout, stderr, command=meta.get("command"), analysis=analysis
        )
        resp["failure_cause"] = failure["cause"]
        resp["model"] = failure.get("model")
        resp["failure_category"] = failure["category"]
        resp["retryable"] = failure["retryable"]
        resp["failure_action"] = failure["action"]
        resp["failure_detail"] = _format_failure_info(failure)
        if failure.get("stdout_diagnostics"):
            resp["stdout_diagnostics"] = failure["stdout_diagnostics"]
        if failure.get("stderr_diagnostics"):
            resp["stderr_diagnostics"] = failure["stderr_diagnostics"]
        # Work the agent had already emitted before the turn failed. Never the
        # "result" -- but the caller needs it, because it is the difference
        # between a clean retry and one that lands on a half-edited tree.
        if analysis.get("last_assistant_text"):
            resp["partial_output"] = analysis["last_assistant_text"]
    return resp


def _wait_tasks(
    task_ids: List[str],
    timeout: Optional[int] = None,
    request_id: Any = None,
    progress_token: Any = None,
) -> Dict[str, Any]:
    """
    Block until all tasks complete, timeout, or request is cancelled.

    Emits the same progress heartbeat as the sync path. This was missing until
    1.9.0, which left the one code path guaranteed to run long as the only one
    that sent nothing at all: stdio servers are subject to a 30 minute idle
    timeout (Claude Code 2.1.203+, they were exempt before), and this tool's
    own default timeout is 1800s. A wait on any Codex run longer than half an
    hour was racing the client's idle timer with zero bytes on the wire.
    """
    deadline = time.time() + timeout if timeout is not None else None
    results = {}
    started = time.time()
    last_progress = started

    while True:
        # Check if the MCP client cancelled this request
        if request_id is not None and _is_cancelled(request_id):
            for tid in task_ids:
                if tid not in results:
                    state = _resolve_task_state(tid)
                    meta = state.get("meta", {}) if isinstance(state, dict) else {}
                    results[tid] = {
                        "status": "cancelled",
                        "task_id": tid,
                        "error": "Wait cancelled by client",
                    }
                    if meta.get("worktree_path"):
                        results[tid]["worktree_path"] = meta["worktree_path"]
                    if meta.get("worktree_branch"):
                        results[tid]["worktree_branch"] = meta["worktree_branch"]
            break

        pending = []
        for tid in task_ids:
            if tid in results:
                continue
            info = _check_task(tid)
            if info["status"] in ("completed", "failed", "cancelled", "not_found", "error"):
                results[tid] = info
            else:
                pending.append(tid)

        if not pending:
            break

        if deadline is not None and time.time() >= deadline:
            for tid in pending:
                state = _resolve_task_state(tid)
                meta = state.get("meta", {}) if isinstance(state, dict) else {}
                results[tid] = {
                    "status": "timeout",
                    "task_id": tid,
                    "error": "Still running (wait timed out, task NOT killed)",
                }
                if meta.get("worktree_path"):
                    results[tid]["worktree_path"] = meta["worktree_path"]
                if meta.get("worktree_branch"):
                    results[tid]["worktree_branch"] = meta["worktree_branch"]
            break

        now = time.time()
        if now - last_progress >= _PROGRESS_INTERVAL:
            last_progress = now
            # _send_progress appends " running (Ns elapsed)", so the label has
            # to be a bare noun phrase to read as a sentence.
            waiting_on = len(pending)
            _send_progress(
                progress_token,
                now - started,
                f"{waiting_on} Codex task{'s' if waiting_on != 1 else ''}",
            )

        sleep_time = 2.0
        if deadline is not None:
            sleep_time = min(sleep_time, max(0.1, deadline - time.time()))
        time.sleep(sleep_time)

    return results

# ===================================================================
# Tool definitions
# ===================================================================

_CODEX_PROPERTIES = {
    "threadId": {
        "type": "string",
        "description": (
            "Optional session/thread ID from a previous Codex call. When set, "
            "this resumes that conversation with full prior context instead of "
            "starting fresh -- the async equivalent of codex_reply, and the "
            "right choice for long follow-ups."
        ),
    },
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
    "worktree": {
        "type": "boolean",
        "description": (
            "Create an isolated git worktree and branch for this task. "
            "Each task gets its own copy of the repo so parallel tasks "
            "never conflict. The response includes the branch name "
            "(codex-swarm/<task_id>) -- merge it back when done."
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
            "Run a Codex session synchronously. Parameters match the official "
            "Codex MCP tool. This server applies NO timeout of its own, but MCP "
            "clients do -- Claude Code cuts a tool call at roughly 300s by "
            "default -- and when the client gives up, the result is lost even "
            "though the underlying `codex exec` keeps running to completion. "
            "Use codex_async + codex_wait for anything that might run longer "
            "than a couple of minutes, and whenever you want more than one "
            "session at a time. Terminal failures report the model actually "
            "used, the structured provider/CLI cause, a category, and "
            "informational retryability. Stderr is labeled secondary "
            "diagnostics and never replaces the cause; this wrapper does not "
            "retry a turn."
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
            "Launch a Codex task and return immediately with a task_id, so "
            "several can run at once. Use this to FAN OUT, then collect with "
            "a single codex_wait. "
            "CRITICAL: the returned task_id is internal to this server. Your "
            "MCP client does not track it, it will not appear in your task "
            "list, and no notification fires when the work finishes. You must "
            "call codex_wait in the SAME turn -- if the turn ends first, the "
            "task still completes but its result is stranded and the session "
            "is never woken. Do not end your turn on a codex_status check "
            "expecting to be called back, because nothing will call you back. "
            "Pass threadId to RESUME an existing session -- prefer this over "
            "codex_reply for any follow-up expected to run more than a few "
            "minutes, since only the async path survives a client idle timeout. "
            "Set worktree=true to isolate each task in its own git worktree "
            "so parallel tasks don't conflict -- merge the branch back when done."
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
            "Continue a Codex conversation synchronously by providing the "
            "thread/session ID and a follow-up prompt. Uses `codex exec resume` "
            "under the hood. Best for short follow-ups; for anything long-running "
            "use codex_async(threadId=..., prompt=...) instead, which returns a "
            "task_id that survives a client idle timeout. Terminal failures "
            "report the configured model, structured cause, category, and "
            "informational retryability; stderr is secondary diagnostics only "
            "and the wrapper does not retry the turn."
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
            "Works on both running and completed tasks. This is a read-only "
            "peek: it collects nothing and does not keep a task attached to "
            "your session -- only codex_wait does that, so never end a turn on "
            "a status check expecting to be called back. Terminal "
            "turn.completed/turn.failed events take precedence; top-level "
            "error events and error items are treated as non-terminal diagnostics. "
            "A terminal failure includes its structured cause, actual model, "
            "category, and informational retryability. A FAILED verdict here "
            "is not always authoritative: when no unambiguous terminal event "
            "or exit code is available, status is inferred from output and the "
            "response says so explicitly. Verify before treating a reported "
            "failure as proof the work did not land. Inferred completion "
            "verdicts are labeled too."
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
            "all results. Accepts a list of task_ids -- pass every id you "
            "launched in ONE call rather than waiting on them one at a time. "
            "This is the call that actually delivers codex_async results, and "
            "the only one that keeps the work attached to your session. "
            "LET IT RUN LONG. A wait that outlives your client's backgrounding "
            "threshold becomes a tracked background task, and your client "
            "re-invokes you with the results when it settles -- that is the "
            "behaviour you want, not something to avoid. Do NOT set a short "
            "timeout to return before it happens: short waits are raised to a "
            "floor anyway, and ducking under the threshold just makes you "
            "responsible for remembering to come back. Already-finished tasks "
            "return instantly regardless. "
            "If the wait itself times out the task is NOT killed; call "
            "codex_wait again with the same task_ids to resume waiting. Failed "
            "tasks return the model actually used, the structured cause, a "
            "category, and informational retryability. Stderr is labeled "
            "secondary diagnostics and never replaces the cause. The wrapper "
            "does not retry a turn."
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
                        "Max seconds to wait (default: 1800). Leave this alone "
                        "unless you have a specific reason; values below the "
                        "client backgrounding floor are raised to it. The task "
                        "keeps running even if this times out."
                    ),
                },
            },
            "required": ["task_ids"],
        },
    },
    {
        "name": "codex_cancel",
        "description": (
            "Kill a running async Codex task. The process is terminated and "
            "the task is marked as cancelled. Any worktree and partial output "
            "are preserved for inspection. If the task had already failed, "
            "the response includes its structured failure details."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task_id to cancel.",
                },
            },
            "required": ["task_id"],
        },
    },
]

# ===================================================================
# MCP Resources -- static server info for discoverability
# ===================================================================

RESOURCES = [
    {
        "uri": "codex-swarm:///server-info",
        "name": "Server Info",
        "description": "Server version, configuration, and capabilities.",
        "mimeType": "application/json",
    },
    {
        "uri": "codex-swarm:///config",
        "name": "Server Config",
        "description": "Current server-level defaults and flags.",
        "mimeType": "application/json",
    },
    {
        "uri": "codex-swarm:///tasks",
        "name": "Active Tasks",
        "description": (
            "List of all known async tasks and their current state. Failed "
            "entries include model, structured cause, category, and retryability."
        ),
        "mimeType": "application/json",
    },
]


def _read_resource(uri: str) -> Optional[str]:
    """Return resource content as JSON string, or None if unknown URI."""
    if uri == "codex-swarm:///server-info":
        return json.dumps({
            "name": "codex-mcp-swarm",
            "version": __version__,
            "tools": [t["name"] for t in TOOLS],
            "task_dir": str(TASK_DIR),
            "worktree_dir": str(WORKTREE_BASE_DIR),
            "task_max_age_seconds": _TASK_MAX_AGE,
            "log_file": LOG_FILE,
            "log_level": LOG_LEVEL,
            "failure_reporting": {
                "terminal_events": ["turn.completed", "turn.failed"],
                "top_level_error_is_terminal": False,
                "error_item_is_terminal": False,
                "stderr_role": "secondary diagnostics only",
                "wrapper_retries": False,
            },
        }, indent=2)

    if uri == "codex-swarm:///config":
        return json.dumps({
            "server_config": SERVER_CONFIG,
            "server_flags": SERVER_FLAGS,
        }, indent=2)

    if uri == "codex-swarm:///tasks":
        tasks = []
        for meta_file in sorted(TASK_DIR.glob("*.meta")):
            try:
                meta = json.loads(meta_file.read_text())
                task_id = meta.get("task_id", meta_file.stem)
                state = _resolve_task_state(task_id)
                entry = {
                    "task_id": task_id,
                    "status": state.get("status"),
                    "elapsed_seconds": state.get("elapsed_seconds"),
                }
                if state.get("status") in ("completed", "failed"):
                    entry.update({
                        "exit_code": state.get("exit_code"),
                        "exit_code_lost": bool(state.get("exit_code_lost")),
                        "status_inferred": bool(state.get("status_inferred")),
                    })
                for key in ("worktree_path", "worktree_branch", "thread_id"):
                    val = meta.get(key)
                    if val:
                        entry[key] = val
                if state.get("status") == "failed":
                    stdout = _safe_read(TASK_DIR / f"{task_id}.stdout")
                    stderr = _safe_read(TASK_DIR / f"{task_id}.stderr")
                    failure = _build_failure_info(
                        stdout, stderr, command=meta.get("command")
                    )
                    entry.update({
                        "model": failure.get("model"),
                        "failure_cause": failure["cause"],
                        "failure_category": failure["category"],
                        "retryable": failure["retryable"],
                        "failure_action": failure["action"],
                    })
                    if failure.get("stderr_diagnostics"):
                        entry["stderr_diagnostics"] = failure["stderr_diagnostics"]
                tasks.append(entry)
            except Exception:
                continue
        return json.dumps(tasks, indent=2)

    return None


def _cancel_task(task_id: str) -> Dict[str, Any]:
    """Kill a running async task and mark it as cancelled."""
    if not _validate_task_id(task_id):
        return {"status": "error", "error": f"Invalid task ID: {task_id}"}

    meta_file = TASK_DIR / f"{task_id}.meta"
    if not meta_file.exists():
        return {"status": "not_found", "error": f"Task {task_id} not found"}

    try:
        meta = json.loads(meta_file.read_text())
    except Exception as exc:
        return {"status": "error", "error": f"Bad metadata: {exc}"}

    if meta.get("status") in ("completed", "failed", "cancelled"):
        resp: Dict[str, Any] = {
            "status": meta["status"],
            "task_id": task_id,
            "message": f"Task already {meta['status']}",
        }
        if meta["status"] == "failed":
            failure = _check_task(task_id)
            for key in (
                "exit_code",
                "exit_code_lost",
                "status_inferred",
                "model",
                "failure_cause",
                "failure_category",
                "retryable",
                "failure_action",
                "failure_detail",
            ):
                if key in failure:
                    resp[key] = failure[key]
        return resp

    pid = meta.get("pid")
    pid_start_time = meta.get("pid_start_time")

    if pid and _is_alive(pid, expected_start_time=pid_start_time):
        # Terminate the process
        try:
            os.kill(pid, signal.SIGTERM)
            logging.info("Sent SIGTERM to PID %d (task %s)", pid, task_id)
        except ProcessLookupError:
            pass  # already dead
        except Exception as exc:
            logging.warning("Failed to kill PID %d: %s", pid, exc)

        # Give it a moment to exit, then force kill
        time.sleep(0.5)
        if _is_alive(pid, expected_start_time=pid_start_time):
            try:
                os.kill(pid, signal.SIGKILL)
                logging.info("Sent SIGKILL to PID %d (task %s)", pid, task_id)
            except ProcessLookupError:
                pass
            except Exception:
                pass

    # Mark as cancelled in metadata
    meta["status"] = "cancelled"
    meta["completed_at"] = time.time()
    try:
        meta_file.write_text(json.dumps(meta, indent=2))
    except Exception:
        pass

    # Cleanup in-memory tracking
    if pid:
        _ASYNC_PIDS.pop(pid, None)
        _ASYNC_PROCS.pop(pid, None)

    resp: Dict[str, Any] = {
        "status": "cancelled",
        "task_id": task_id,
        "message": "Task cancelled",
    }
    if meta.get("worktree_path"):
        resp["worktree_path"] = meta["worktree_path"]
    if meta.get("worktree_branch"):
        resp["worktree_branch"] = meta["worktree_branch"]
    return resp

# ===================================================================
# Request handler
# ===================================================================

def _handle(request: Dict[str, Any]) -> None:
    method = request.get("method")
    rid = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        requested = params.get("protocolVersion")
        negotiated = (
            requested
            if requested in _SUPPORTED_PROTOCOL_VERSIONS
            else _DEFAULT_PROTOCOL_VERSION
        )
        if requested != negotiated:
            logging.info(
                "Client requested protocol %s; responding with %s",
                requested, negotiated,
            )
        _send({
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": negotiated,
                "capabilities": {"tools": {}, "resources": {}},
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
            # Kill the child immediately rather than waiting for the next
            # 2s poll -- and, more importantly, so it cannot outlive the
            # request as an untracked orphan.
            _terminate_sync_proc(cancelled_id)
        return

    if method == "tools/list":
        _send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
        return

    if method == "resources/list":
        _send({"jsonrpc": "2.0", "id": rid, "result": {"resources": RESOURCES}})
        return

    if method == "resources/read":
        uri = params.get("uri", "")
        content = _read_resource(uri)
        if content is not None:
            _send({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": content,
                    }],
                },
            })
        else:
            _send({
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": -32602, "message": f"Unknown resource: {uri}"},
            })
        return

    if method == "tools/call":
        tool = params.get("name")
        args = params.get("arguments", {})
        # Clients that want progress supply a token here; without one the
        # heartbeat is a no-op and long sync runs stay silent as before.
        progress_token = (params.get("_meta") or {}).get("progressToken")

        if tool == "codex":
            args.pop("timeout", None)  # ignored: sync tools run to completion
            result, thread_id, worktree_info = _run_sync(
                args,
                request_id=rid,
                progress_token=progress_token,
            )
            text = result
            details = []
            if thread_id:
                details.append(
                    f"Thread ID: {thread_id}\nUse codex_reply(threadId=\"{thread_id}\", prompt=\"...\") to continue this session."
                )
            if worktree_info:
                details.append(
                    f"Worktree Path: {worktree_info['worktree_path']}\n"
                    f"Worktree Branch: {worktree_info['worktree_branch']}"
                )
            if details:
                text += f"\n\n---\n" + "\n\n---\n".join(details)
            _send({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {"content": [{"type": "text", "text": text}]},
            })

        elif tool == "codex_async":
            launch_info = _start_async(args)
            task_id = launch_info["task_id"]
            lines = [
                f"Codex task {task_id} started.",
            ]
            if launch_info.get("worktree_path"):
                lines.append(f"Worktree Path: {launch_info['worktree_path']}")
            if launch_info.get("worktree_branch"):
                lines.append(f"Worktree Branch: {launch_info['worktree_branch']}")
            # This warning exists because the old wording ("started in
            # background") read exactly like the MCP client's own backgrounded
            # tool calls, which DO notify on completion. A session conflated the
            # two, ended its turn, and never collected a finished result.
            lines.extend([
                "",
                "NOT TRACKED BY YOUR CLIENT. This id is internal to "
                "codex-swarm. No completion notification will arrive and "
                "nothing will wake this session when the task finishes.",
                "",
                f'You MUST call codex_wait(task_ids=["{task_id}"]) before this '
                "turn ends, or the result is stranded on disk with nobody to "
                "collect it. codex_wait is the call your client tracks and the "
                "only thing that brings the answer back.",
                "",
                f'codex_status(task_ids=["{task_id}"]) is a progress peek only. '
                "It does not collect the result and it does not keep the "
                "session alive.",
            ])
            _send({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": "\n".join(lines),
                    }],
                },
            })

        elif tool == "codex_reply":
            thread_id = args.get("threadId")
            prompt = args.get("prompt")

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
                thread_id, prompt, request_id=rid, progress_token=progress_token
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
                stderr_path = TASK_DIR / f"{tid}.stderr"
                stdout_text = _safe_read(stdout_path)
                jsonl_status = _parse_jsonl_status(stdout_path)

                if status == "running":
                    lines = [f"=== Task {tid} ({elapsed}s elapsed) ==="]
                elif status == "failed":
                    exit_code = state.get("exit_code")
                    if exit_code is None:
                        exit_code = "?"
                    lines = [f"=== Task {tid} (FAILED in {elapsed}s, exit {exit_code}) ==="]
                    if state.get("status_inferred"):
                        lines.append(_UNVERIFIED_FAILURE_HINT)
                elif status == "cancelled":
                    lines = [f"=== Task {tid} (CANCELLED after {elapsed}s) ==="]
                else:
                    lines = [f"=== Task {tid} (COMPLETED in {elapsed}s) ==="]
                    if state.get("status_inferred"):
                        lines.append(_UNVERIFIED_COMPLETION_HINT)

                # Surface thread_id from meta or stdout
                meta = state.get("meta", {})
                tid_thread = meta.get("thread_id")
                if not tid_thread and stdout_text:
                    _, tid_thread = _extract_from_jsonl(stdout_text)
                if tid_thread:
                    lines.append(f"Thread ID: {tid_thread}")
                if meta.get("worktree_path"):
                    lines.append(f"Worktree: {meta['worktree_path']}")
                if meta.get("worktree_branch"):
                    lines.append(f"Branch: {meta['worktree_branch']}")

                display_phase = (
                    status
                    if status in ("completed", "failed", "cancelled")
                    else jsonl_status["phase"]
                )
                lines.append(f"Phase: {display_phase}")
                lines.append(f"Tools called: {jsonl_status['tools_called']}")

                if status == "failed":
                    failure = _build_failure_info(
                        stdout_text,
                        _safe_read(stderr_path),
                        command=meta.get("command"),
                    )
                    lines.extend(
                        _format_failure_info(failure).splitlines()
                    )
                elif status == "running" and jsonl_status["last_error"]:
                    observed = _classify_failure(jsonl_status["last_error"])
                    retryable = observed.get("retryable")
                    retryable_text = (
                        "yes"
                        if retryable is True
                        else "no"
                        if retryable is False
                        else "unknown"
                    )
                    lines.append(
                        "Error observed (non-terminal; process still running): "
                        f"{jsonl_status['last_error'][:300]}"
                    )
                    lines.append(f"Category: {observed['category']}")
                    lines.append(
                        "Retryable: "
                        f"{retryable_text} (informational only; the wrapper did not retry)"
                    )

                if jsonl_status["last_warning"]:
                    lines.append(f"Warning: {jsonl_status['last_warning'][:300]}")

                if jsonl_status["terminal_conflict"]:
                    lines.append(
                        "Terminal signals conflicted; lifecycle state used the process exit code."
                    )

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
                    output_label = (
                        "Partial output (before failure)"
                        if status == "failed"
                        else "Output"
                    )
                    lines.append(
                        f"{output_label}: {jsonl_status['last_assistant_text'][:300]}"
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

            # Raise short waits past the client's backgrounding threshold.
            # Callers pick values like 110 or 115 specifically to return before
            # the client backgrounds the call, which is exactly backwards: the
            # backgrounded call is the one whose completion re-invokes the
            # session. Ducking under it converts a guaranteed wake-up into a
            # promise the model has to remember to keep. Finished tasks are
            # unaffected -- they resolve before the deadline is consulted.
            # `timeout: null` is left alone deliberately: that means "no
            # deadline", which already outlives the threshold.
            if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
                if timeout < _WAIT_MIN_TIMEOUT:
                    logging.info(
                        "Raising codex_wait timeout %ss -> %ss to cross the "
                        "client's %ss auto-background threshold",
                        timeout, _WAIT_MIN_TIMEOUT, _CLIENT_AUTO_BACKGROUND_S,
                    )
                    timeout = _WAIT_MIN_TIMEOUT

            results = _wait_tasks(
                task_ids,
                timeout=timeout,
                request_id=rid,
                progress_token=progress_token,
            )

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
                    if info.get("worktree_path"):
                        header += f"\nWorktree: {info['worktree_path']}"
                    if info.get("worktree_branch"):
                        header += f"\nBranch: {info['worktree_branch']}"
                    detail = info["result"]
                    if info.get("status_inferred"):
                        detail = f"{_UNVERIFIED_COMPLETION_HINT}\n\n{detail}"
                    parts.append(f"{header}\n{detail}")
                elif info["status"] == "failed":
                    exit_code = info.get("exit_code")
                    if exit_code is None:
                        exit_code = "?"
                    header = (
                        f"=== Task {tid} (FAILED in "
                        f"{info['elapsed_seconds']}s, exit {exit_code}) ==="
                    )
                    if info.get("worktree_path"):
                        header += f"\nWorktree: {info['worktree_path']}"
                    if info.get("worktree_branch"):
                        header += f"\nBranch: {info['worktree_branch']}"
                    detail = info.get("failure_detail") or (
                        "Model: unknown\n"
                        f"Cause: {info.get('failure_cause', 'Unknown Codex failure')}"
                    )
                    if info.get("partial_output"):
                        detail += (
                            "\n\nPartial output (before failure): "
                            f"{info['partial_output'][:300]}"
                        )
                    if info.get("status_inferred"):
                        detail = f"{_UNVERIFIED_FAILURE_HINT}\n\n{detail}"
                    parts.append(f"{header}\n{detail}")
                elif info["status"] == "cancelled":
                    header = (
                        f"=== Task {tid} (CANCELLED after "
                        f"{info['elapsed_seconds']}s) ==="
                    )
                    if info.get("worktree_path"):
                        header += f"\nWorktree: {info['worktree_path']}"
                    if info.get("worktree_branch"):
                        header += f"\nBranch: {info['worktree_branch']}"
                    detail = info.get("result", "Task was cancelled")
                    parts.append(f"{header}\n{detail}")
                elif info["status"] == "timeout":
                    header = (
                        f"=== Task {tid} === STILL RUNNING (wait timed out, "
                        f"task is NOT killed -- call codex_wait again to "
                        f"resume waiting)"
                    )
                    if info.get("worktree_path"):
                        header += f"\nWorktree: {info['worktree_path']}"
                    if info.get("worktree_branch"):
                        header += f"\nBranch: {info['worktree_branch']}"
                    parts.append(f"{header}\n{_STILL_RUNNING_HINT}")
                else:
                    header = (
                        f"=== Task {tid} === "
                        f"{info.get('error', info['status'])}"
                    )
                    if info.get("worktree_path"):
                        header += f"\nWorktree: {info['worktree_path']}"
                    if info.get("worktree_branch"):
                        header += f"\nBranch: {info['worktree_branch']}"
                    parts.append(header)

            _send({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "content": [{"type": "text", "text": "\n\n".join(parts)}],
                },
            })

        elif tool == "codex_cancel":
            task_id = args.get("task_id")
            if not task_id:
                _send({
                    "jsonrpc": "2.0",
                    "id": rid,
                    "error": {"code": -32602, "message": "task_id is required"},
                })
                return

            result = _cancel_task(task_id)
            lines = [f"Task {task_id}: {result.get('message', result.get('error', result['status']))}"]
            if result.get("worktree_path"):
                lines.append(f"Worktree preserved: {result['worktree_path']}")
            if result.get("worktree_branch"):
                lines.append(f"Branch preserved: {result['worktree_branch']}")
            if result.get("failure_detail"):
                lines.extend(["", result["failure_detail"]])
            _send({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "content": [{"type": "text", "text": "\n".join(lines)}],
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
    finally:
        # Async tasks are detached on purpose (start_new_session) and survive;
        # sync children belong to a request that no longer exists.
        _terminate_all_sync_procs()

    logging.info("Server stopped")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
codex-mcp-swarm -- Parallel Codex MCP Server

An MCP server that wraps OpenAI's Codex CLI with true parallel execution,
live task monitoring, and full parameter parity with the official Codex
MCP tool. Built on stdlib-only ``mcp-swarm-core``.

Features:
  - codex:        Synchronous execution (drop-in replacement; returns an MCP Task
                  when the client supports io.modelcontextprotocol/tasks)
  - codex_async:  Launch a task, return a task_id immediately (fan-out).
                  NOT fire-and-forget: the id is server-side only, so the
                  result reaches the caller solely via codex_wait.
  - codex_reply:  Continue a previous Codex session (with MCP Tasks support)
  - codex_status: Live view of what each task is doing
  - codex_wait:   Block until multiple tasks complete
  - codex_cancel: Kill a running async task and its process tree

Protocol support:
  - MCP 2026-07-28 stateless discovery (`server/discover`), per-request protocol
    version/capability metadata, and the official MCP Tasks extension
    (`io.modelcontextprotocol/tasks`: `CreateTaskResult`, `tasks/get`,
    `tasks/cancel`).
  - Backward-compatible stateful `initialize` negotiation (2024-11-05 through
    2025-11-25), top-level `instructions`, `ping`, tool `annotations`,
    `outputSchema` + `structuredContent`, `isError`, and enriched
    `notifications/progress`.

Server-level defaults are set via -c key=value CLI args, matching the
exact same format as `codex mcp-server` for drop-in config compatibility.

Originally inspired by jeanchristophe13v/codex-mcp-async.
Rewritten with full flag parity, live JSONL status parsing, batch wait,
and session reply support.

License: MIT
Repository: https://github.com/TKasperczyk/codex-mcp-swarm
"""

import argparse
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import mcp_swarm_core as _core
except ImportError as _exc:
    raise ImportError(
        "codex-mcp-swarm requires 'mcp-swarm-core'. Install via "
        "'pip install codex-mcp-swarm' or run via 'uvx codex-mcp-swarm'."
    ) from _exc

__version__ = "1.14.0"

_IS_WINDOWS = _core.IS_WINDOWS
_DEFAULT_TMP = _core.DEFAULT_TMP
_DEFAULT_BASE = Path(_core.default_user_tmp_dir("codex-swarm"))
_DEFAULT_WORKTREE_DIR = Path(_core.default_worktree_dir("codex-swarm"))
_TEXT_KW = _core.TEXT_KW

# ---------------------------------------------------------------------------
# Logging (configurable via env vars)
# ---------------------------------------------------------------------------
_LOG_ENV = os.environ.get("CODEX_SWARM_LOG")
LOG_FILE = _LOG_ENV or str(_DEFAULT_BASE / "codex_mcp_swarm.log")
LOG_LEVEL = os.environ.get("CODEX_SWARM_LOG_LEVEL", "WARNING").upper()

_core.prepare_log_parent(
    LOG_FILE, is_default=_LOG_ENV is None, default_base=_DEFAULT_BASE
)
logging.basicConfig(
    filename=LOG_FILE,
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Task storage
# ---------------------------------------------------------------------------
_TASK_DIR_ENV = os.environ.get("CODEX_SWARM_TASK_DIR")
TASK_DIR = _core.prepare_runtime_dir(
    Path(_TASK_DIR_ENV) if _TASK_DIR_ENV else (_DEFAULT_BASE / "tasks"),
    is_default=_TASK_DIR_ENV is None,
    default_base=_DEFAULT_BASE,
)

_WORKTREE_DIR_ENV = os.environ.get("CODEX_SWARM_WORKTREE_DIR")
WORKTREE_BASE_DIR = _core.prepare_runtime_dir(
    Path(_WORKTREE_DIR_ENV) if _WORKTREE_DIR_ENV else _DEFAULT_WORKTREE_DIR,
    is_default=_WORKTREE_DIR_ENV is None,
    default_base=_DEFAULT_WORKTREE_DIR.parent,
)

# ---------------------------------------------------------------------------
# Protocol & runtime constants (re-exported for compatibility)
# ---------------------------------------------------------------------------
_STATELESS_PROTOCOL_VERSIONS = _core.STATELESS_PROTOCOL_VERSIONS
_SUPPORTED_PROTOCOL_VERSIONS = _core.SUPPORTED_PROTOCOL_VERSIONS
_DEFAULT_PROTOCOL_VERSION = _core.DEFAULT_PROTOCOL_VERSION
_ALL_PROTOCOL_VERSIONS = _core.ALL_PROTOCOL_VERSIONS
_TASKS_EXTENSION_ID = _core.TASKS_EXTENSION_ID
_DISCOVER_TTL_MS = _core.DISCOVER_TTL_MS
_TASK_POLL_INTERVAL_MS = _core.TASK_POLL_INTERVAL_MS
_BLOCKING_METHODS = _core.BLOCKING_METHODS

# Server-level config (populated in main from CLI args)
SERVER_CONFIG: Dict[str, str] = {}
SERVER_FLAGS: List[str] = []

# Re-export shared SIGCHLD async process tracking
_NOT_REAPED = _core.NOT_REAPED
_ASYNC_PIDS = _core.ASYNC_PIDS
_ASYNC_PROCS = _core.ASYNC_PROCS
_sigchld_handler = _core.sigchld_handler

# Polling, progress, wait floor, and cleanup defaults
_POLL_INTERVAL = 2  # seconds between cancellation/timeout checks
_PROGRESS_INTERVAL = 20  # seconds between heartbeat progress notifications
_CLIENT_AUTO_BACKGROUND_S = 120
_WAIT_MIN_TIMEOUT = _core.parse_env_int(
    "CODEX_SWARM_MIN_WAIT", _CLIENT_AUTO_BACKGROUND_S + 30
)
_TASK_MAX_AGE = _core.parse_env_int("CODEX_SWARM_TASK_MAX_AGE", 86400)
_last_cleanup = 0.0

_PROCESS_CHECK_CMD = (
    '`tasklist /FI "IMAGENAME eq codex.exe"`'
    if _IS_WINDOWS
    else "`pgrep -af 'codex exec'`"
)
_HINTS = _core.make_inferred_hints("Codex", "codex_wait", _PROCESS_CHECK_CMD)
_UNVERIFIED_FAILURE_HINT = _HINTS["unverified_failure_hint"]
_UNVERIFIED_COMPLETION_HINT = _HINTS["unverified_completion_hint"]
_STILL_RUNNING_HINT = _HINTS["still_running_hint"]

# ---------------------------------------------------------------------------
# Platform wrappers (forwarding live module globals for testability)
# ---------------------------------------------------------------------------
_win_open_process = _core.win_open_process
_win_is_running = _core.win_is_running
_win_start_time = _core.win_start_time
_win_system32 = _core.win_system32
_win_kill_tree = _core.win_kill_tree
_posix_signal_tree = _core.posix_signal_tree
_posix_group_alive = _core.posix_group_alive
_validate_task_id = _core.validate_task_id
_iso_utc = _core.iso_utc
_clean_none = _core.clean_none
_error_message = _core.error_message
_decide_run_status = _core.decide_run_status
_format_task_activity_brief = _core.format_task_activity_brief
_failure_structured = _core.failure_structured
_is_valid_request_id = _core.is_valid_request_id
_client_supports_tasks = _core.client_supports_tasks
_validate_task_ids_param = _core.validate_task_ids_param


def _popen_kw(detach: bool = False, new_group: bool = False) -> Dict[str, Any]:
    return _core.popen_kw(detach=detach, new_group=new_group, is_windows=_IS_WINDOWS)


def _stop_proc(proc: subprocess.Popen, own_group: Optional[bool] = None) -> None:
    _core.stop_proc(
        proc,
        own_group=own_group,
        is_windows=_IS_WINDOWS,
        win_kill_tree_fn=lambda pid: _win_kill_tree(pid),
    )


def _get_pid_start_time(pid: int) -> Optional[float]:
    return _core.get_pid_start_time(pid, is_windows=_IS_WINDOWS)


def _is_alive(
    pid: Optional[int], expected_start_time: Optional[float] = None
) -> bool:
    return _core.is_alive(
        pid,
        expected_start_time=expected_start_time,
        is_windows=_IS_WINDOWS,
        get_start_time_fn=lambda p: _get_pid_start_time(p),
    )


def _safe_read(path: Path) -> str:
    return _core.safe_read(path)


def _write_meta_atomic(meta_file: Path, meta: Dict[str, Any]) -> None:
    _core.write_meta_atomic(meta_file, meta, is_windows=_IS_WINDOWS)


def _prompt_stdin(prompt: Optional[str], stdin_path: Optional[Path] = None):
    return _core.prompt_stdin(
        prompt,
        stdin_path=stdin_path,
        task_dir=TASK_DIR,
        is_windows=_IS_WINDOWS,
    )


def _create_worktree(run_id: str, base_cwd: Optional[str]) -> Tuple[str, str, str]:
    return _core.create_worktree(
        run_id,
        base_cwd,
        worktree_base_dir=WORKTREE_BASE_DIR,
        branch_prefix="codex-swarm",
        popen_kw_fn=lambda **kw: _popen_kw(**kw),
    )


def _remove_worktree(worktree_path: str, branch_name: str) -> bool:
    return _core.remove_worktree(
        worktree_path,
        branch_name,
        worktree_base_dir=WORKTREE_BASE_DIR,
        popen_kw_fn=lambda **kw: _popen_kw(**kw),
    )


# ---------------------------------------------------------------------------
# Codex CLI launcher (Windows npm shim avoidance)
# ---------------------------------------------------------------------------
_codex_launcher_cache: Optional[List[str]] = None


def _codex_launcher() -> List[str]:
    """argv prefix that runs the Codex CLI.

    POSIX: plain ``codex``. Windows: resolves npm's ``codex.cmd`` shim to
    ``node codex.js`` (or native ``codex.exe``) without ever touching cmd.exe
    or executing untrusted binaries from the current working directory.
    ``CODEX_SWARM_CODEX_BIN`` overrides lookup with a ``.exe`` or ``.js`` path.
    """
    global _codex_launcher_cache
    if not _IS_WINDOWS:
        return ["codex"]
    if _codex_launcher_cache is not None:
        return _codex_launcher_cache

    def node_for(js: str, near: Optional[str] = None) -> List[str]:
        local_node = os.path.join(near, "node.exe") if near else None
        node = (
            os.path.abspath(local_node)
            if local_node and os.path.isfile(local_node)
            else _core.safe_which("node", is_windows=True)
        )
        if not node:
            raise RuntimeError(f"node.exe not found (needed to run {js})")
        return [node, js]

    override = os.environ.get("CODEX_SWARM_CODEX_BIN")
    if override:
        override = os.path.abspath(override)
        if override.lower().endswith(".js"):
            launcher = node_for(override)
        elif override.lower().endswith(".exe"):
            launcher = [override]
        else:
            raise RuntimeError(
                f"CODEX_SWARM_CODEX_BIN={override} must be a .exe or .js file; "
                "batch files would route the prompt through cmd.exe"
            )
    else:
        found = _core.safe_which("codex", is_windows=True)
        if not found:
            raise RuntimeError("codex not found on PATH (set CODEX_SWARM_CODEX_BIN)")
        found = os.path.abspath(found)
        if found.lower().endswith(".exe"):
            launcher = [found]
        else:
            shim_dir = os.path.dirname(found)
            if os.path.basename(shim_dir).lower() == ".bin":
                modules_dir = os.path.dirname(shim_dir)
            else:
                modules_dir = os.path.join(shim_dir, "node_modules")
            js = os.path.join(modules_dir, "@openai", "codex", "bin", "codex.js")
            if not os.path.isfile(js):
                raise RuntimeError(
                    f"{found} is a script shim and {js} does not exist; "
                    "refusing to run Codex through cmd.exe (set CODEX_SWARM_CODEX_BIN)"
                )
            launcher = node_for(js, near=shim_dir)

    logging.info("Codex launcher: %s", launcher)
    _codex_launcher_cache = launcher
    return launcher


def _exec_argv(cmd: List[str]) -> List[str]:
    """Map a logical ``codex ...`` command to the actual executable argv."""
    if cmd and cmd[0] == "codex":
        return _codex_launcher() + cmd[1:]
    return cmd


# ---------------------------------------------------------------------------
# Codex JSONL parser (single-pass + incremental cache via mcp-swarm-core)
# ---------------------------------------------------------------------------
def _new_jsonl_state() -> Dict[str, Any]:
    return {
        "thread_id": None,
        "last_assistant_text": None,
        "errors": [],
        "warnings": [],
        "terminal_events": [],
        "terminal_error": None,
        "saw_json": False,
        "saw_turn_started": False,
        "phase": "starting",
        "tools_called": 0,
        "last_tool": None,
        "last_tool_args": None,
        "last_reasoning": None,
        "last_assistant_preview": None,
    }


def _clone_jsonl_state(state: Dict[str, Any]) -> Dict[str, Any]:
    cloned = dict(state)
    cloned["errors"] = list(state["errors"])
    cloned["warnings"] = list(state["warnings"])
    cloned["terminal_events"] = list(state["terminal_events"])
    return cloned


def _consume_jsonl_lines(state: Dict[str, Any], lines: List[str]) -> None:
    """Parse Codex exec/session JSONL lines in a single pass."""
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        state["saw_json"] = True
        etype = event.get("type", "")

        if etype == "thread.started":
            tid = event.get("thread_id")
            if isinstance(tid, str) and tid:
                state["thread_id"] = tid
        elif etype == "session_meta":
            payload = event.get("payload", {})
            if isinstance(payload, dict) and not state["thread_id"]:
                tid = payload.get("id")
                if isinstance(tid, str) and tid:
                    state["thread_id"] = tid

        if etype == "turn.started":
            state["saw_turn_started"] = True
            state["phase"] = "running"
        elif etype == "turn.completed":
            state["terminal_events"].append("completed")
        elif etype == "turn.failed":
            state["terminal_events"].append("failed")
            message = _error_message(event.get("error"))
            if message:
                state["terminal_error"] = message
        elif etype == "error":
            message = _error_message(event.get("message")) or _error_message(
                event.get("error")
            )
            if message:
                state["errors"].append(message)
        elif etype in ("item.started", "item.completed"):
            item = event.get("item", {})
            if isinstance(item, dict):
                itype = item.get("type", "")
                if itype == "command_execution":
                    if etype == "item.started":
                        state["tools_called"] += 1
                        state["last_tool"] = "exec_command"
                        cmd_val = item.get("command", "")
                        state["last_tool_args"] = (
                            cmd_val[:150]
                            if isinstance(cmd_val, str)
                            else str(cmd_val)[:150]
                        )
                    elif etype == "item.completed":
                        output = item.get("aggregated_output", "")
                        if isinstance(output, str) and output:
                            state["last_reasoning"] = output[-200:]
                elif itype == "mcp_tool_call" and etype == "item.started":
                    state["tools_called"] += 1
                    tool_name = item.get("tool") or item.get("name") or "mcp_tool"
                    state["last_tool"] = str(tool_name)
                    tool_args = item.get("arguments")
                    if tool_args is not None:
                        state["last_tool_args"] = str(tool_args)[:150]
                elif itype == "file_change" and etype == "item.completed":
                    state["tools_called"] += 1
                    state["last_tool"] = "file_change"
                    changes = item.get("changes")
                    if isinstance(changes, list) and changes:
                        first = changes[0]
                        if isinstance(first, dict) and first.get("path"):
                            state["last_tool_args"] = str(first["path"])[:150]
                elif itype == "reasoning" and etype == "item.completed":
                    text_val = item.get("text", "")
                    if isinstance(text_val, str) and text_val:
                        state["last_reasoning"] = text_val[-200:]
                elif itype == "agent_message" and etype == "item.completed":
                    text_val = item.get("text", "")
                    if isinstance(text_val, str) and text_val:
                        state["last_assistant_text"] = text_val
                        state["last_assistant_preview"] = text_val[-300:]
                elif itype == "error" and etype == "item.completed":
                    message = _error_message(item.get("message")) or _error_message(
                        item.get("error")
                    )
                    if message:
                        state["warnings"].append(message)
        elif etype == "response_item":
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                continue
            ptype = payload.get("type", "")
            if ptype == "function_call":
                state["tools_called"] += 1
                raw_name = payload.get("name")
                state["last_tool"] = (
                    str(raw_name) if raw_name is not None else ""
                ) or "?"
                raw_args = payload.get("arguments", "{}")
                try:
                    args = (
                        json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    )
                    if isinstance(args, dict):
                        chosen = (
                            args.get("cmd")
                            or args.get("command")
                            or args.get("path")
                            or args.get("pattern")
                        )
                        if chosen is not None:
                            state["last_tool_args"] = (
                                chosen[:150]
                                if isinstance(chosen, str)
                                else str(chosen)[:150]
                            )
                        else:
                            state["last_tool_args"] = str(args)[:150]
                    else:
                        state["last_tool_args"] = str(args)[:150]
                except (json.JSONDecodeError, TypeError):
                    state["last_tool_args"] = str(raw_args)[:150]
            elif ptype == "reasoning":
                content = payload.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            text_val = c.get("text", "")
                            if isinstance(text_val, str) and text_val:
                                state["last_reasoning"] = text_val[-200:]
            elif payload.get("role") == "assistant" and ptype == "message":
                content = payload.get("content")
                if isinstance(content, list):
                    texts = [
                        c["text"]
                        for c in content
                        if isinstance(c, dict)
                        and c.get("type") in ("output_text", "text", "input_text")
                        and isinstance(c.get("text"), str)
                    ]
                    if texts:
                        joined = "\n".join(texts)
                        state["last_assistant_text"] = joined
                        state["last_assistant_preview"] = texts[-1][-300:]
        elif etype == "event_msg":
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                continue
            ptype = payload.get("type")
            if ptype == "task_started":
                state["saw_turn_started"] = True
                state["phase"] = "running"
            elif ptype == "task_complete":
                state["terminal_events"].append("completed")
                message = payload.get("last_agent_message")
                if isinstance(message, str) and message:
                    state["last_assistant_text"] = message
                    state["last_assistant_preview"] = message[-300:]
            elif ptype == "turn_aborted":
                state["terminal_events"].append("failed")
                reason = payload.get("reason")
                if isinstance(reason, str) and reason:
                    state["terminal_error"] = f"Turn aborted: {reason}"
            elif ptype == "error":
                message = _error_message(payload.get("message")) or _error_message(
                    payload.get("error")
                )
                if message:
                    state["errors"].append(message)


def _finalize_jsonl_views(
    state: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    terminal_events = list(state["terminal_events"])
    terminal_types = set(terminal_events)
    terminal_status: Optional[str] = None
    terminal_conflict = False
    if len(terminal_types) == 1:
        terminal_status = next(iter(terminal_types))
    elif len(terminal_types) > 1:
        terminal_conflict = True

    errors = list(state["errors"])
    warnings = list(state["warnings"])

    analysis: Dict[str, Any] = {
        "thread_id": state["thread_id"],
        "last_assistant_text": state["last_assistant_text"],
        "errors": errors,
        "warnings": warnings,
        "terminal_events": terminal_events,
        "terminal_status": terminal_status,
        "terminal_error": state["terminal_error"],
        "terminal_conflict": terminal_conflict,
        "saw_json": state["saw_json"],
        "saw_turn_started": state["saw_turn_started"],
    }

    phase = state["phase"]
    if terminal_status:
        phase = terminal_status
    elif terminal_conflict:
        phase = "unknown (conflicting terminal events)"

    status: Dict[str, Any] = {
        "phase": phase,
        "tools_called": state["tools_called"],
        "last_tool": state["last_tool"],
        "last_tool_args": state["last_tool_args"],
        "last_reasoning": state["last_reasoning"],
        "last_assistant_text": state["last_assistant_preview"],
        "last_error": errors[-1] if errors else None,
        "last_warning": warnings[-1] if warnings else None,
        "terminal_error": state["terminal_error"],
        "terminal_conflict": terminal_conflict,
    }
    return analysis, status


_JSONL_CACHE_MIN_BYTES = 8192
_JSONL_CACHE_MAX_ENTRIES = 128
_JSONL_CACHE_OBJ = _core.IncrementalJsonlCache(
    _core.JsonlParserAdapter(
        new_state=_new_jsonl_state,
        clone_state=_clone_jsonl_state,
        consume_lines=_consume_jsonl_lines,
        finalize_views=_finalize_jsonl_views,
    ),
    min_bytes=_JSONL_CACHE_MIN_BYTES,
    max_entries=_JSONL_CACHE_MAX_ENTRIES,
)
_JSONL_CACHE = _JSONL_CACHE_OBJ.entries


def _invalidate_jsonl_cache(stdout_path: Path) -> None:
    _JSONL_CACHE_OBJ.invalidate(stdout_path)


def _parse_jsonl_full(jsonl_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _JSONL_CACHE_OBJ.parse_text(jsonl_text)


def _analyze_jsonl(jsonl_text: str) -> Dict[str, Any]:
    analysis, _ = _parse_jsonl_full(jsonl_text)
    return analysis


def _parse_jsonl_file(stdout_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _JSONL_CACHE_OBJ.parse_file(
        stdout_path,
        open_fn=lambda *a, **kw: open(*a, **kw),
    )


def _parse_jsonl_status(stdout_path: Path) -> Dict[str, Any]:
    _, status = _parse_jsonl_file(stdout_path)
    return status


def _extract_thread_id_fast(
    stdout_path: Path, meta: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    if meta and isinstance(meta.get("thread_id"), str) and meta["thread_id"]:
        return meta["thread_id"]
    if not stdout_path.exists():
        return None
    try:
        with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
            bytes_read = 0
            for line in f:
                bytes_read += len(line)
                stripped = line.strip()
                if stripped:
                    try:
                        event = json.loads(stripped)
                    except json.JSONDecodeError:
                        event = None
                    if isinstance(event, dict):
                        etype = event.get("type", "")
                        if etype == "thread.started":
                            tid = event.get("thread_id")
                            if isinstance(tid, str) and tid:
                                return tid
                        if etype == "session_meta":
                            payload = event.get("payload", {})
                            if isinstance(payload, dict):
                                tid = payload.get("id")
                                if isinstance(tid, str) and tid:
                                    return tid
                if bytes_read >= 65536:
                    break
    except OSError:
        return None
    return None


def _extract_result(
    stdout: str, stderr: str, analysis: Optional[Dict[str, Any]] = None
) -> Tuple[str, Optional[str]]:
    analysis = analysis or _analyze_jsonl(stdout)
    if analysis.get("saw_json") and analysis.get("last_assistant_text"):
        return analysis["last_assistant_text"], analysis.get("thread_id")
    result = stdout.strip()
    if not result and stderr:
        result = stderr.strip()
    return result or "No output from Codex", analysis.get("thread_id")


def _extract_from_jsonl(jsonl_text: str) -> Tuple[Optional[str], Optional[str]]:
    analysis = _analyze_jsonl(jsonl_text)
    return analysis.get("last_assistant_text"), analysis.get("thread_id")


# ---------------------------------------------------------------------------
# Codex model parsing, failure classification, and command builder
# ---------------------------------------------------------------------------
_CODEX_VALUE_FLAGS = frozenset({
    "-c", "--config",
    "-m", "--model",
    "-s", "--sandbox",
    "-p", "--profile",
    "-C", "--cd",
    "-i", "--image",
    "-o", "--output-last-message",
    "--enable", "--disable",
    "--remote", "--remote-auth-token-env",
    "--local-provider",
    "--add-dir",
    "--thread-source",
    "--cyber-access-program",
    "--output-schema",
    "--color",
})


def _strip_toml_quotes(value: str) -> Optional[str]:
    val = value.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
        val = val[1:-1]
    return val or None


def _flatten_config(prefix: str, value: Any, out: Dict[str, str]) -> None:
    """Flatten nested config values into TOML-compatible dotted key/value pairs."""
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten_config(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, bool):
        out[prefix] = "true" if value else "false"
    elif isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, (dict, list)):
                logging.warning(
                    "Skipping unsupported nested %s in config list %s",
                    type(item).__name__,
                    prefix,
                )
                continue
            if isinstance(item, str):
                escaped = (
                    item.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("\n", "\\n")
                )
                parts.append(f'"{escaped}"')
            elif isinstance(item, bool):
                parts.append("true" if item else "false")
            else:
                parts.append(str(item))
        out[prefix] = f"[{', '.join(parts)}]"
    else:
        out[prefix] = str(value)


def _resolve_model(params: Optional[dict] = None) -> Optional[str]:
    params = params or {}
    model = params.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    per_call = params.get("config")
    if isinstance(per_call, dict):
        flat: Dict[str, str] = {}
        for k, v in per_call.items():
            _flatten_config(str(k), v, flat)
        cfg_model = flat.get("model")
        if isinstance(cfg_model, str) and cfg_model.strip():
            return _strip_toml_quotes(cfg_model)
    srv_model = SERVER_CONFIG.get("model")
    if isinstance(srv_model, str) and srv_model.strip():
        return _strip_toml_quotes(srv_model)
    return None


def _model_from_command(command: Any) -> Optional[str]:
    if isinstance(command, (list, tuple)):
        tokens = [str(t) for t in command]
    elif isinstance(command, str) and command.strip():
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
    else:
        return None

    i = 0
    for idx, tok in enumerate(tokens[:4]):
        if tok == "exec":
            i = idx + 1
            break
    if i < len(tokens) and tokens[i] == "resume":
        i += 1

    flag_model: Optional[str] = None
    config_model: Optional[str] = None

    while i < len(tokens):
        tok = tokens[i]
        if tok == "--" or tok == "-" or not tok.startswith("-"):
            break
        if tok in ("-m", "--model"):
            if i + 1 < len(tokens):
                flag_model = _strip_toml_quotes(tokens[i + 1])
                i += 2
                continue
            break
        if tok.startswith("--model="):
            flag_model = _strip_toml_quotes(tok.split("=", 1)[1])
            i += 1
            continue
        if tok in ("-c", "--config"):
            if i + 1 < len(tokens):
                cfg = tokens[i + 1]
                if cfg.startswith("model="):
                    config_model = _strip_toml_quotes(cfg.split("=", 1)[1])
                i += 2
                continue
            break
        if tok.startswith("--config="):
            cfg = tok.split("=", 1)[1]
            if cfg.startswith("model="):
                config_model = _strip_toml_quotes(cfg.split("=", 1)[1])
            i += 1
            continue
        if tok in _CODEX_VALUE_FLAGS:
            i += 2
            continue
        i += 1

    return flag_model or config_model


def _classify_failure(message: str) -> Dict[str, Any]:
    return _core.classify_failure(message, product_name="Codex")


def _build_failure_info(
    stdout: str,
    stderr: str,
    command: Any = None,
    analysis: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    analysis = analysis or _analyze_jsonl(stdout)
    return _core.build_failure_info(
        stdout,
        stderr,
        analysis=analysis,
        product_name="Codex",
        command=command,
        model=model,
        model_from_command_fn=_model_from_command,
        classify_fn=_classify_failure,
    )


def _format_failure_info(
    info: Dict[str, Any], include_diagnostics: bool = True
) -> str:
    return _core.format_failure_info(
        info, include_diagnostics=include_diagnostics, product_name="Codex"
    )


def _with_json_flag(cmd: List[str]) -> List[str]:
    if "--json" in cmd:
        return cmd
    idx = 3 if len(cmd) > 2 and cmd[2] == "resume" else 2
    return cmd[:idx] + ["--json"] + cmd[idx:]


def _build_command(params: dict) -> Tuple[List[str], Optional[str]]:
    cmd = ["codex", "exec"]

    thread_id = params.get("threadId")
    is_resume = bool(thread_id)
    if is_resume:
        cmd.append("resume")

    merged = dict(SERVER_CONFIG)
    per_call = params.get("config") or {}
    if isinstance(per_call, dict):
        for k, v in per_call.items():
            _flatten_config(str(k), v, merged)

    model = params.get("model")
    if model:
        cmd.extend(["-m", model])
        merged.pop("model", None)
    elif "model" in merged:
        raw_model = _strip_toml_quotes(str(merged.pop("model")))
        if raw_model:
            cmd.extend(["-m", raw_model])

    sandbox = params.get("sandbox")
    if sandbox:
        merged["sandbox_mode"] = sandbox
    if not is_resume and "sandbox_mode" in merged:
        raw_sandbox = _strip_toml_quotes(str(merged.pop("sandbox_mode")))
        if raw_sandbox:
            cmd.extend(["-s", raw_sandbox])

    approval = params.get("approval-policy")
    if approval:
        merged["approval_policy"] = approval

    profile = params.get("profile")
    if profile:
        if is_resume:
            logging.warning(
                "Resume of %s: dropping profile=%r "
                "(codex exec resume has no --profile flag)",
                thread_id,
                profile,
            )
        else:
            cmd.extend(["-p", profile])

    cwd = params.get("cwd")
    if cwd and not is_resume:
        cmd.extend(["-C", cwd])

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
        tid_str = str(thread_id)
        if tid_str.startswith("-"):
            cmd.append("--")
        cmd.append(tid_str)

    prompt = params.get("prompt", "")
    if prompt:
        cmd.append("-")

    if is_resume:
        proc_cwd = os.path.abspath(cwd) if cwd else None
    else:
        proc_cwd = cwd if cwd and os.path.isabs(cwd) else None
    return cmd, proc_cwd


def _build_reply_command(
    thread_id: str,
    prompt: str,
    extra_params: Optional[dict] = None,
) -> List[str]:
    params: Dict[str, Any] = dict(extra_params or {})
    params["threadId"] = thread_id
    params["prompt"] = prompt
    cmd, _ = _build_command(params)
    return cmd


def _validate_optional_config(raw: Any) -> Optional[str]:
    if raw is not None and not isinstance(raw, dict):
        return "config must be an object"
    return None


def _validate_optional_thread_id(raw: Any) -> Optional[str]:
    if raw is not None and (not isinstance(raw, str) or not raw.strip()):
        return "threadId must be a non-empty string"
    return None


def _validate_codex_params(
    args: Dict[str, Any], *, is_reply: bool = False
) -> Optional[str]:
    err = _validate_optional_config(args.get("config"))
    if err:
        return err
    if not is_reply:
        err = _validate_optional_thread_id(args.get("threadId"))
        if err:
            return err
        wt = args.get("worktree")
        if wt is not None and not isinstance(wt, bool):
            return "worktree must be a boolean"
    str_keys = (
        ("cwd", "model", "sandbox", "approval-policy")
        if is_reply
        else (
            "cwd",
            "model",
            "sandbox",
            "approval-policy",
            "profile",
            "base-instructions",
            "developer-instructions",
            "compact-prompt",
        )
    )
    for key in str_keys:
        val = args.get(key)
        if val is not None and not isinstance(val, str):
            return f"{key} must be a string"
    return None


# ---------------------------------------------------------------------------
# Server instructions, Tool definitions & Resources
# ---------------------------------------------------------------------------
SERVER_INSTRUCTIONS = (
    "codex-mcp-swarm wraps the Codex CLI for parallel execution, session resume, "
    "and live progress monitoring.\n\n"
    "Execution lifecycle:\n"
    "- Sync calls (`codex`, `codex_reply`) have no server timeout, but MCP clients "
    "typically enforce ~300s limits; use `codex_async` + `codex_wait` for long runs "
    "or parallel fan-out. Clients supporting MCP 2026-07-28 with "
    "`io.modelcontextprotocol/tasks` receive native async tasks directly from "
    "`codex` and `codex_reply` (poll via `tasks/get`, cancel via `tasks/cancel`).\n"
    "- `codex_async` task IDs are internal to this server and NOT tracked by legacy "
    "MCP clients. Always call `codex_wait` in the same turn (passing all launched "
    "task_ids in one call). `codex_status` is a read-only peek and collects nothing.\n"
    "- Let `codex_wait` run past the client's ~120s auto-backgrounding threshold; "
    "short timeouts are raised to a floor (finished tasks still return instantly). "
    "A wait timeout is NOT a failure and does not kill tasks -- call `codex_wait` again.\n\n"
    "Failure reporting:\n"
    "- Terminal `turn.completed`/`turn.failed` JSONL events take precedence; top-level "
    "`error` events and `item.completed` error items are non-terminal diagnostics. "
    "Failures report the actual model, structured cause, category, and informational "
    "retryability. Stderr is labeled secondary diagnostics and never replaces the cause; "
    "this wrapper never retries a turn.\n"
    "- When neither a terminal event nor an observed exit code is available, the verdict "
    "is explicitly flagged as inferred and should be verified before acting on it.\n\n"
    "Worktree isolation:\n"
    "- `worktree: true` creates an isolated git worktree and branch (`codex-swarm/<id>`) "
    "based on `HEAD`. Uncommitted and untracked working-tree changes are NOT included "
    "in the worktree -- commit them first if the Codex task needs them."
)

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
        "description": "The user prompt for the Codex session.",
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
            "Create an isolated git worktree and branch (codex-swarm/<task_id>) "
            "based on HEAD so parallel tasks never conflict. Note: branches from "
            "HEAD and excludes uncommitted or untracked working-tree changes -- "
            "commit them first if the task needs them. Merge the branch back when done."
        ),
    },
    "model": {
        "type": "string",
        "description": "Optional override for the model name (e.g. 'gpt-6.2-astra').",
    },
    "profile": {
        "type": "string",
        "description": "Configuration profile from config.toml (ignored on resume).",
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

_FAILURE_SCHEMA = _core.FAILURE_SCHEMA
_CODEX_OUTPUT_SCHEMA = _core.make_sync_output_schema("thread_id")
_CODEX_ASYNC_OUTPUT_SCHEMA = _core.make_async_output_schema()
_CODEX_REPLY_OUTPUT_SCHEMA = _core.make_reply_output_schema("thread_id")
_CODEX_STATUS_OUTPUT_SCHEMA = _core.make_status_output_schema("thread_id")
_CODEX_WAIT_OUTPUT_SCHEMA = _core.make_wait_output_schema("thread_id")
_CODEX_CANCEL_OUTPUT_SCHEMA = _core.make_cancel_output_schema()

TOOLS = [
    {
        "name": "codex",
        "description": (
            "Run a Codex session synchronously (or as a native MCP Task when "
            "the client advertises io.modelcontextprotocol/tasks). Parameters "
            "match the official Codex MCP tool. This server applies NO timeout "
            "of its own, but legacy MCP clients typically cut a tool call around "
            "300s -- use codex_async + codex_wait for long runs or parallel "
            "fan-out on non-Tasks clients."
        ),
        "inputSchema": {
            "type": "object",
            "properties": _CODEX_PROPERTIES,
            "required": ["prompt"],
        },
        "outputSchema": _CODEX_OUTPUT_SCHEMA,
        "annotations": {
            "title": "Run Codex",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
    {
        "name": "codex_async",
        "description": (
            "Launch a Codex task and return immediately with a task_id for "
            "parallel fan-out, then collect with a single codex_wait. "
            "CRITICAL: the task_id is internal to this server and NOT tracked "
            "by your MCP client -- you MUST call codex_wait in the same turn "
            "or results are stranded on disk. Pass threadId to resume an "
            "existing session asynchronously, and set worktree=true to isolate "
            "parallel edits in a HEAD-based git worktree."
        ),
        "inputSchema": {
            "type": "object",
            "properties": _CODEX_PROPERTIES,
            "required": ["prompt"],
        },
        "outputSchema": _CODEX_ASYNC_OUTPUT_SCHEMA,
        "annotations": {
            "title": "Launch Async Codex Task",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
    {
        "name": "codex_reply",
        "description": (
            "Continue a Codex conversation via `codex exec resume` (runs "
            "synchronously, or as a native MCP Task when the client advertises "
            "io.modelcontextprotocol/tasks). For long follow-ups on non-Tasks "
            "clients, prefer codex_async(threadId=..., prompt=...) + codex_wait."
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
                "cwd": _CODEX_PROPERTIES["cwd"],
                "model": _CODEX_PROPERTIES["model"],
                "sandbox": _CODEX_PROPERTIES["sandbox"],
                "approval-policy": _CODEX_PROPERTIES["approval-policy"],
                "config": _CODEX_PROPERTIES["config"],
            },
            "required": ["prompt", "threadId"],
        },
        "outputSchema": _CODEX_REPLY_OUTPUT_SCHEMA,
        "annotations": {
            "title": "Continue Codex Session",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
    {
        "name": "codex_status",
        "description": (
            "Get live status of async Codex tasks (last tool call, reasoning, "
            "warnings, and terminal failure details). This is a read-only peek: "
            "it collects nothing and does not keep a task attached to your "
            "session -- only codex_wait does that. A FAILED verdict here is not "
            "always authoritative when inferred without a terminal event or "
            "exit code (flagged explicitly in the output)."
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
        "outputSchema": _CODEX_STATUS_OUTPUT_SCHEMA,
        "annotations": {
            "title": "Check Async Codex Task Status",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "codex_wait",
        "description": (
            "Block until one or more async Codex tasks complete, then return "
            "all results. Pass every task_id you launched in ONE call. Let it "
            "run long past your client's backgrounding threshold (short "
            "timeouts are raised to a floor; finished tasks return instantly). "
            "A wait timeout does NOT kill tasks -- call codex_wait again to "
            "keep waiting."
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
        "outputSchema": _CODEX_WAIT_OUTPUT_SCHEMA,
        "annotations": {
            "title": "Wait for Async Codex Tasks",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "codex_cancel",
        "description": (
            "Kill a running async Codex task and its child processes. Any "
            "worktree and partial output are preserved for inspection. If the "
            "task had already failed, the response includes its structured "
            "failure details."
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
        "outputSchema": _CODEX_CANCEL_OUTPUT_SCHEMA,
        "annotations": {
            "title": "Cancel Async Codex Task",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
]

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


# ---------------------------------------------------------------------------
# Backend adapter & server wiring
# ---------------------------------------------------------------------------
class CodexBackend(_core.SwarmBackend):
    server_name = "codex-mcp-swarm"
    version = __version__
    product_name = "Codex"
    tool_prefix = "codex"
    uri_scheme = "codex-swarm"
    branch_prefix = "codex-swarm"
    session_param_name = "threadId"
    session_field_name = "thread_id"
    session_label = "Thread ID"
    prefer_observed_session_id = False
    process_check_cmd = _PROCESS_CHECK_CMD
    server_instructions = SERVER_INSTRUCTIONS
    tools = TOOLS
    resources = RESOURCES
    reply_extra_param_keys = ("cwd", "model", "sandbox", "approval-policy", "config")

    def validate_call_params(
        self, args: Dict[str, Any], *, is_reply: bool = False
    ) -> Optional[str]:
        return _validate_codex_params(args, is_reply=is_reply)

    def build_command(self, params: Dict[str, Any]) -> Tuple[List[str], Optional[str]]:
        return _build_command(params)

    def prepare_command_for_json(self, cmd: List[str]) -> List[str]:
        return _with_json_flag(cmd)

    def exec_argv(self, cmd: List[str]) -> List[str]:
        return _exec_argv(cmd)

    def resolve_model(self, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        return _resolve_model(params)

    def model_from_command(self, command: Any) -> Optional[str]:
        return _model_from_command(command)

    def parse_jsonl_file(
        self, stdout_path: Path
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return _parse_jsonl_file(stdout_path)

    def parse_jsonl_text(
        self, jsonl_text: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return _parse_jsonl_full(jsonl_text)

    def invalidate_jsonl_cache(self, stdout_path: Path) -> None:
        _invalidate_jsonl_cache(stdout_path)

    def extract_session_id_fast(
        self, stdout_path: Path, meta: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        return _extract_thread_id_fast(stdout_path, meta)

    def extract_result(
        self,
        stdout: str,
        stderr: str,
        analysis: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[str]]:
        return _extract_result(stdout, stderr, analysis=analysis)

    def decide_run_status(
        self,
        exit_code: Optional[int],
        analysis: Dict[str, Any],
        stderr: str,
    ) -> Tuple[str, bool]:
        return _decide_run_status(exit_code, analysis, stderr)

    def classify_failure(self, message: str) -> Dict[str, Any]:
        return _classify_failure(message)

    def build_failure_info(
        self,
        stdout: str,
        stderr: str,
        command: Any = None,
        analysis: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        return _build_failure_info(
            stdout, stderr, command=command, analysis=analysis, model=model
        )

    def format_failure_info(
        self, info: Dict[str, Any], include_diagnostics: bool = True
    ) -> str:
        return _format_failure_info(info, include_diagnostics=include_diagnostics)

    def get_server_info_extra(self) -> Dict[str, Any]:
        return {
            "failure_reporting": {
                "terminal_events": ["turn.completed", "turn.failed"],
                "top_level_error_is_terminal": False,
                "error_item_is_terminal": False,
                "stderr_role": "secondary diagnostics only",
                "wrapper_retries": False,
            },
        }

    def get_config_resource(self) -> Dict[str, Any]:
        return {
            "server_config": SERVER_CONFIG,
            "server_flags": SERVER_FLAGS,
        }


def _set_last_cleanup(val: float) -> None:
    global _last_cleanup
    _last_cleanup = val


_HOOKS = _core.RuntimeHooks(
    task_dir=lambda: TASK_DIR,
    worktree_base_dir=lambda: WORKTREE_BASE_DIR,
    branch_prefix="codex-swarm",
    task_max_age=lambda: _TASK_MAX_AGE,
    wait_min_timeout=lambda: _WAIT_MIN_TIMEOUT,
    client_auto_background_s=_CLIENT_AUTO_BACKGROUND_S,
    poll_interval=lambda: _POLL_INTERVAL,
    progress_interval=lambda: _PROGRESS_INTERVAL,
    log_file=lambda: LOG_FILE,
    log_level=lambda: LOG_LEVEL,
    is_windows_fn=lambda: _IS_WINDOWS,
    popen_kw_fn=lambda **kw: _popen_kw(**kw),
    get_pid_start_time_fn=lambda pid: _get_pid_start_time(pid),
    is_alive_fn=lambda pid, expected_start_time=None: _is_alive(
        pid, expected_start_time=expected_start_time
    ),
    win_kill_tree_fn=lambda pid: _win_kill_tree(pid),
    stop_proc_fn=lambda proc, own_group=None: _stop_proc(proc, own_group=own_group),
    write_meta_atomic_fn=lambda mf, meta: _write_meta_atomic(mf, meta),
    safe_read_fn=lambda path: _safe_read(path),
    create_worktree_fn=lambda run_id, base_cwd: _create_worktree(run_id, base_cwd),
    remove_worktree_fn=lambda wt_path, branch: _remove_worktree(wt_path, branch),
    send_fn=lambda resp: _send(resp),
    start_async_fn=lambda args: _start_async(args),
    get_last_cleanup_fn=lambda: _last_cleanup,
    set_last_cleanup_fn=_set_last_cleanup,
)

_BACKEND = CodexBackend()
_SERVER = _core.McpSwarmServer(_BACKEND, _HOOKS)

# ---------------------------------------------------------------------------
# Compatibility exports (preserving module-level API & test monkeypatch hooks)
# ---------------------------------------------------------------------------
_SYNC_PROCS = _SERVER.sync_procs.procs
_sync_procs_lock = _SERVER.sync_procs.lock
_cancelled_requests = _SERVER.cancellations.cancelled_requests
_cancelled_lock = _SERVER.cancellations.lock


def _mark_cancelled(rid: Any) -> None:
    _SERVER.cancellations.mark_cancelled(rid)


def _is_cancelled(rid: Any) -> bool:
    return _SERVER.cancellations.is_cancelled(rid)


def _send(response: Dict[str, Any]) -> None:
    _SERVER.send(response)


def _send_progress(
    progress_token: Any,
    elapsed: float,
    label: str,
    detail: Optional[str] = None,
) -> None:
    _SERVER.send_progress(progress_token, elapsed, label, detail)


def _register_sync_proc(request_id: Any, proc: subprocess.Popen) -> None:
    _SERVER.sync_procs.register(request_id, proc)


def _unregister_sync_proc(request_id: Any) -> None:
    _SERVER.sync_procs.unregister(request_id)


def _terminate_sync_proc(request_id: Any) -> bool:
    return _SERVER.sync_procs.terminate(
        request_id,
        is_valid_id_fn=_is_valid_request_id,
        stop_proc_fn=_stop_proc,
    )


def _terminate_all_sync_procs() -> None:
    _SERVER.sync_procs.terminate_all(
        is_windows=_IS_WINDOWS,
        win_kill_tree_fn=lambda pid: _win_kill_tree(pid),
    )


def _cleanup_old_tasks() -> None:
    _SERVER.tasks.cleanup_old_tasks()


def _start_async_internal(
    params: dict,
    *,
    high_entropy_id: bool = False,
    mcp_task: bool = False,
    tool_name: str = "codex_async",
) -> Dict[str, Any]:
    return _SERVER.tasks.start_async(
        params,
        high_entropy_id=high_entropy_id,
        mcp_task=mcp_task,
        tool_name=tool_name,
    )


def _start_async(params: dict) -> Dict[str, Optional[str]]:
    return _start_async_internal(params)


def _resolve_task_state(task_id: str) -> Dict[str, Any]:
    return _SERVER.tasks.resolve_task_state(task_id)


def _check_task(
    task_id: str, state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return _SERVER.tasks.check_task(task_id, state=state)


def _wait_tasks(
    task_ids: List[str],
    timeout: Optional[float] = None,
    request_id: Any = None,
    progress_token: Any = None,
) -> Dict[str, Any]:
    return _SERVER.tasks.wait_tasks(
        task_ids,
        timeout=timeout,
        request_id=request_id,
        progress_token=progress_token,
    )


def _cancel_task(task_id: str) -> Dict[str, Any]:
    return _SERVER.tasks.cancel_task(task_id)


def _wait_proc_detailed(
    proc: subprocess.Popen,
    deadline: Optional[float] = None,
    request_id: Any = None,
    timeout_msg: str = "Error: Codex execution timed out",
    progress_token: Any = None,
    progress_label: str = "Codex",
    command: Any = None,
    model: Optional[str] = None,
    own_group: Optional[bool] = None,
) -> Dict[str, Any]:
    return _SERVER.tasks.wait_proc_detailed(
        proc,
        deadline=deadline,
        request_id=request_id,
        timeout_msg=timeout_msg,
        progress_token=progress_token,
        progress_label=progress_label,
        command=command,
        model=model,
        own_group=own_group,
    )


def _wait_proc(
    proc: subprocess.Popen,
    deadline: Optional[float] = None,
    request_id: Any = None,
    timeout_msg: str = "Error: Codex execution timed out",
    progress_token: Any = None,
    progress_label: str = "Codex",
    command: Any = None,
    model: Optional[str] = None,
) -> Tuple[str, Optional[str], bool]:
    outcome = _wait_proc_detailed(
        proc,
        deadline=deadline,
        request_id=request_id,
        timeout_msg=timeout_msg,
        progress_token=progress_token,
        progress_label=progress_label,
        command=command,
        model=model,
    )
    return outcome["text"], outcome["thread_id"], outcome["failed"]


def _run_sync_detailed(
    params: dict,
    request_id: Any = None,
    progress_token: Any = None,
) -> Dict[str, Any]:
    return _SERVER.tasks.run_sync_detailed(
        params,
        request_id=request_id,
        progress_token=progress_token,
    )


def _run_sync(
    params: dict,
    request_id: Any = None,
    progress_token: Any = None,
) -> Tuple[str, Optional[str], Optional[Dict[str, str]]]:
    outcome = _run_sync_detailed(
        params, request_id=request_id, progress_token=progress_token
    )
    return outcome["text"], outcome["thread_id"], outcome.get("worktree_info")


def _run_reply_sync_detailed(
    thread_id: str,
    prompt: str,
    extra_params: Optional[dict] = None,
    request_id: Any = None,
    progress_token: Any = None,
) -> Dict[str, Any]:
    return _SERVER.tasks.run_reply_sync_detailed(
        thread_id,
        prompt,
        extra_params=extra_params,
        request_id=request_id,
        progress_token=progress_token,
    )


def _run_reply_sync(
    thread_id: str,
    prompt: str,
    request_id: Any = None,
    progress_token: Any = None,
) -> Tuple[str, Optional[str]]:
    outcome = _run_reply_sync_detailed(
        thread_id, prompt, request_id=request_id, progress_token=progress_token
    )
    return outcome["text"], outcome["thread_id"]


def _read_resource(uri: str) -> Optional[str]:
    return _SERVER.read_resource(uri)


def _handle(request: Dict[str, Any]) -> None:
    _SERVER.handle(request)


def _handle_threaded(request: Dict[str, Any]) -> None:
    _SERVER.handle_threaded(request)


# ---------------------------------------------------------------------------
# Main CLI entrypoint
# ---------------------------------------------------------------------------
def _parse_args() -> None:
    parser = argparse.ArgumentParser(
        description="codex-mcp-swarm -- Parallel Codex MCP Server",
        usage="codex-mcp-swarm [-c key=value]... [--skip-git-repo-check] [--ephemeral]",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
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
    _SERVER.run_stdio_loop(
        on_start=lambda: logging.info(
            "Server starting -- defaults=%s flags=%s", SERVER_CONFIG, SERVER_FLAGS
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
codex-mcp-swarm -- Parallel Codex MCP Server

An MCP server that wraps OpenAI's Codex CLI with true parallel execution,
live task monitoring, and full parameter parity with the official Codex
MCP tool.

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

import sys
import json
import subprocess
import os
import time
import math
import logging
import signal
import secrets
import traceback
import argparse
import contextlib
import re
import shlex
import shutil
import tempfile
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

__version__ = "1.13.0"

_IS_WINDOWS = sys.platform == "win32"

# /tmp does not exist on Windows; use the per-user temp dir there.
_DEFAULT_TMP = tempfile.gettempdir() if _IS_WINDOWS else "/tmp"

# ---------------------------------------------------------------------------
# Logging (configurable via env vars)
# ---------------------------------------------------------------------------
LOG_FILE = os.environ.get(
    "CODEX_SWARM_LOG", os.path.join(_DEFAULT_TMP, "codex_mcp_swarm.log")
)
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
TASK_DIR = Path(
    os.environ.get("CODEX_SWARM_TASK_DIR", os.path.join(_DEFAULT_TMP, "codex_swarm_tasks"))
)
TASK_DIR.mkdir(exist_ok=True, mode=0o700)

WORKTREE_BASE_DIR = Path(
    os.environ.get(
        "CODEX_SWARM_WORKTREE_DIR", os.path.join(_DEFAULT_TMP, "codex-swarm-worktrees")
    )
)
WORKTREE_BASE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

# ---------------------------------------------------------------------------
# Platform layer (Windows)
# ---------------------------------------------------------------------------
# Everything POSIX-specific stays inline where it always was; Windows gets its
# own branch through these helpers so the Linux/macOS path is unchanged.

# Subprocess text decoding. Codex and git emit UTF-8; Windows would otherwise
# decode with the ANSI code page and mangle or reject non-ASCII output.
_TEXT_KW: Dict[str, Any] = (
    {"text": True, "encoding": "utf-8", "errors": "replace"}
    if _IS_WINDOWS
    else {"text": True}
)

if _IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    _kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _kernel32.GetProcessTimes.restype = wintypes.BOOL
    _kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.GetSystemDirectoryW.restype = wintypes.UINT
    _kernel32.GetSystemDirectoryW.argtypes = [wintypes.LPWSTR, wintypes.UINT]

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _STILL_ACTIVE = 259
    _ERROR_ACCESS_DENIED = 5

    # No console window per child (the server itself usually has none, so
    # each console child would pop one up) and its own process group so a
    # console Ctrl+C aimed at the server does not reach detached tasks.
    _POPEN_FLAGS = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP


def _win_open_process(pid: int) -> Tuple[Optional[int], bool]:
    """Return (handle, access_denied). handle is None if the PID is gone."""
    handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if handle:
        return handle, False
    return None, ctypes.get_last_error() == _ERROR_ACCESS_DENIED


def _win_is_running(pid: int) -> bool:
    handle, denied = _win_open_process(pid)
    if handle is None:
        return denied  # exists but not ours to inspect -- same as POSIX EPERM
    try:
        code = wintypes.DWORD()
        if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        _kernel32.CloseHandle(handle)


def _win_start_time(pid: int) -> Optional[float]:
    handle, _ = _win_open_process(pid)
    if handle is None:
        return None
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not _kernel32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            return None
        created = times[0]
        return ((created.dwHighDateTime << 32) | created.dwLowDateTime) / 1e7
    finally:
        _kernel32.CloseHandle(handle)


def _win_system32(exe: str) -> str:
    """Absolute path to a System32 binary.

    A bare name would go through CreateProcess's search order, which checks
    the server's current directory (often a repository) before System32.
    """
    buf = ctypes.create_unicode_buffer(260)
    if not _kernel32.GetSystemDirectoryW(buf, len(buf)):
        raise OSError(ctypes.get_last_error(), "GetSystemDirectoryW failed")
    return os.path.join(buf.value, exe)


def _win_kill_tree(pid: int) -> bool:
    """Kill a process and every descendant. Returns False if any survived.

    Codex on Windows runs as node -> codex.exe -> tool shells, and killing
    only the top process orphans the rest (which also keep our stdout pipes
    open, so communicate() would never return).
    """
    try:
        result = subprocess.run(
            [_win_system32("taskkill.exe"), "/T", "/F", "/PID", str(pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
            timeout=30,
        )
    except Exception as exc:
        logging.warning("taskkill failed for PID %d: %s", pid, exc)
        return False
    # taskkill exits non-zero when any process in the tree could not be
    # killed (e.g. access denied), so judge by whether the root is gone.
    if result.returncode != 0:
        logging.warning("taskkill exited %d for PID %d", result.returncode, pid)
    return result.returncode == 0 or not _win_is_running(pid)


_codex_launcher_cache: Optional[List[str]] = None


def _codex_launcher() -> List[str]:
    """argv prefix that runs the Codex CLI.

    POSIX: plain ``codex``. Windows: npm installs ``codex.cmd``, and running a
    batch file routes every argument through cmd.exe, which truncates the
    prompt at the first newline and interprets ``& | < > ^ %``. So resolve the
    shim to ``node codex.js`` (or a native codex.exe) and never touch cmd.exe.
    ``CODEX_SWARM_CODEX_BIN`` overrides the lookup with an .exe or .js path;
    anything else (notably .cmd/.bat) is refused for the same reason.
    """
    global _codex_launcher_cache
    if not _IS_WINDOWS:
        return ["codex"]
    if _codex_launcher_cache is not None:
        return _codex_launcher_cache

    def node_for(js: str, near: Optional[str] = None) -> List[str]:
        local_node = os.path.join(near, "node.exe") if near else None
        node = local_node if local_node and os.path.isfile(local_node) else shutil.which("node")
        if not node:
            raise RuntimeError(f"node.exe not found (needed to run {js})")
        return [node, js]

    override = os.environ.get("CODEX_SWARM_CODEX_BIN")
    if override:
        # Absolute now: Codex runs with cwd set to the task's directory or
        # worktree, where a relative path would name a different file.
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
        found = shutil.which("codex")
        if not found:
            raise RuntimeError("codex not found on PATH (set CODEX_SWARM_CODEX_BIN)")
        if found.lower().endswith(".exe"):
            launcher = [found]
        else:
            shim_dir = os.path.dirname(found)
            # Global npm puts the shim beside node_modules; a project-local
            # install puts it in node_modules\.bin.
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
    """Map a logical ``codex ...`` command to what actually gets executed.

    Commands stay ``["codex", ...]`` everywhere else (logging, metadata,
    index-based flag insertion); only the spawn sees the resolved launcher.
    """
    if cmd and cmd[0] == "codex":
        return _codex_launcher() + cmd[1:]
    return cmd


def _popen_kw(detach: bool = False, new_group: bool = False) -> Dict[str, Any]:
    """Platform-specific Popen kwargs for subprocess children."""
    if _IS_WINDOWS:
        return {"creationflags": _POPEN_FLAGS}
    return {"start_new_session": True} if (detach or new_group) else {}


def _posix_signal_tree(pid: int, sig: int, own_group: bool = False) -> bool:
    """Send ``sig`` to ``pid``'s process group when it leads its own group, else ``pid``.

    Returns True when the signal was delivered or the target was already gone.
    Guards against signalling PID <= 1 or the server's own process group.
    """
    if not isinstance(pid, int) or pid <= 1:
        return False
    my_pgrp = os.getpgrp()
    use_pgid: Optional[int] = None
    try:
        pgid = os.getpgid(pid)
        if pgid == pid and pgid != my_pgrp:
            use_pgid = pgid
    except ProcessLookupError:
        if own_group and pid != my_pgrp:
            use_pgid = pid
        else:
            return True
    except Exception:
        pass

    if use_pgid is not None:
        try:
            os.killpg(use_pgid, sig)
            return True
        except ProcessLookupError:
            return True
        except Exception as exc:
            logging.warning(
                "Failed to signal process group %d with %s: %s", use_pgid, sig, exc
            )

    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return True
    except Exception as exc:
        logging.warning("Failed to signal PID %d with %s: %s", pid, sig, exc)
        return False


def _posix_group_alive(pgid: int) -> bool:
    """Return True if any process remains in POSIX process group ``pgid``."""
    if not isinstance(pgid, int) or pgid <= 1 or pgid == os.getpgrp():
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _stop_proc(proc: subprocess.Popen, own_group: Optional[bool] = None) -> None:
    """Terminate a child process tree, escalating to kill after 5s, and reap it."""
    if _IS_WINDOWS:
        _win_kill_tree(proc.pid)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return

    my_pgrp = os.getpgrp()
    if own_group is None:
        in_own_group = False
        try:
            in_own_group = os.getpgid(proc.pid) == proc.pid and proc.pid != my_pgrp
        except Exception:
            in_own_group = False
    else:
        in_own_group = bool(own_group) and proc.pid > 1 and proc.pid != my_pgrp

    if in_own_group:
        _posix_signal_tree(proc.pid, signal.SIGTERM, own_group=True)
    else:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if in_own_group:
            _posix_signal_tree(proc.pid, signal.SIGKILL, own_group=True)
        else:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        proc.wait()

    # If the group leader exited on SIGTERM while grandchildren survived in
    # its process group, sweep the remaining group members with SIGKILL.
    if in_own_group and _posix_group_alive(proc.pid):
        time.sleep(0.05)
        if _posix_group_alive(proc.pid):
            _posix_signal_tree(proc.pid, signal.SIGKILL, own_group=True)

# ---------------------------------------------------------------------------
# Protocol negotiation
# ---------------------------------------------------------------------------
# Legacy stateful revisions (negotiated via `initialize`), newest first.
# Stateless 2026-07-28 replaces `initialize` with `server/discover` and
# per-request `_meta["io.modelcontextprotocol/protocolVersion"]`.
_STATELESS_PROTOCOL_VERSIONS = (
    "2026-07-28",
)
_SUPPORTED_PROTOCOL_VERSIONS = (
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
_DEFAULT_PROTOCOL_VERSION = _SUPPORTED_PROTOCOL_VERSIONS[0]
_ALL_PROTOCOL_VERSIONS = _STATELESS_PROTOCOL_VERSIONS + _SUPPORTED_PROTOCOL_VERSIONS
_TASKS_EXTENSION_ID = "io.modelcontextprotocol/tasks"
_DISCOVER_TTL_MS = 3600000
_TASK_POLL_INTERVAL_MS = 2000

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
    # Strictly lock-free: never invoke logging or anything that acquires locks
    # inside a POSIX signal handler, or SIGCHLD arriving mid-log can deadlock.
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
        except ChildProcessError:
            # Already reaped by subprocess internals -- exit code lost
            _ASYNC_PIDS[pid] = None
        except OSError:
            pass


# Windows has no SIGCHLD and no zombies: Popen.poll() on the kept-alive
# handle in _ASYNC_PROCS returns the real exit code without any reaping race.
if hasattr(signal, "SIGCHLD"):
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
            **_TEXT_KW,
            **_popen_kw(),
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
            **_TEXT_KW,
            **_popen_kw(),
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
            **_TEXT_KW,
            **_popen_kw(),
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
            **_TEXT_KW,
            **_popen_kw(),
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
                **_TEXT_KW,
                **_popen_kw(),
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
                **_TEXT_KW,
                **_popen_kw(),
                check=True,
            )
            logging.info("Deleted worktree branch %s", branch_name)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            logging.debug("Ignoring worktree branch delete failure for %s: %s", branch_name, detail)

    return True


def _get_pid_start_time(pid: int) -> Optional[float]:
    """Get process start time from /proc (Linux). Returns None if unavailable."""
    if _IS_WINDOWS:
        return _win_start_time(pid)
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
    if _IS_WINDOWS:
        # os.kill(pid, 0) is not a probe on Windows: any signal other than
        # CTRL_C/CTRL_BREAK calls TerminateProcess and would kill the task.
        if not _win_is_running(pid):
            return False
    else:
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
_CANCELLED_MAX_ENTRIES = 1024
_cancelled_requests: Set[Any] = set()
_cancelled_order: Deque[Any] = deque()
_cancelled_lock = threading.Lock()


def _is_valid_request_id(rid: Any) -> bool:
    """JSON-RPC 2.0 request IDs must be strings or finite numbers (never bool/list/dict)."""
    if isinstance(rid, bool):
        return False
    if isinstance(rid, (str, int)):
        return True
    if isinstance(rid, float):
        return not (math.isnan(rid) or math.isinf(rid))
    return False


def _mark_cancelled(rid: Any) -> None:
    if not _is_valid_request_id(rid):
        return
    with _cancelled_lock:
        if rid in _cancelled_requests:
            return
        while (
            len(_cancelled_requests) >= _CANCELLED_MAX_ENTRIES
            or len(_cancelled_order) >= _CANCELLED_MAX_ENTRIES * 2
        ) and _cancelled_order:
            oldest = _cancelled_order.popleft()
            _cancelled_requests.discard(oldest)
        _cancelled_requests.add(rid)
        _cancelled_order.append(rid)


def _is_cancelled(rid: Any) -> bool:
    if not _is_valid_request_id(rid):
        return False
    with _cancelled_lock:
        return rid in _cancelled_requests


def _send(response: Dict[str, Any]) -> None:
    # Don't send responses for cancelled requests
    rid = response.get("id")
    if rid is not None and _is_valid_request_id(rid) and _is_cancelled(rid):
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
            safe_rid = rid if _is_valid_request_id(rid) else None
            err = {
                "jsonrpc": "2.0",
                "id": safe_rid,
                "error": {"code": -32603, "message": f"Serialization error: {exc}"},
            }
            print(json.dumps(err), flush=True)


# Accepts both legacy 8-char hex task IDs (codex_async) and 32-char
# high-entropy hex task IDs (MCP Tasks extension io.modelcontextprotocol/tasks).
_TASK_ID_RE = re.compile(r"^[0-9a-f]{8}(?:[0-9a-f]{24})?$")


def _validate_task_id(task_id: Any) -> bool:
    """Reject task IDs that aren't our generated 8-char or 32-char hex format."""
    return isinstance(task_id, str) and bool(_TASK_ID_RE.match(task_id))


def _iso_utc(ts: float) -> str:
    """Format a POSIX timestamp as an ISO 8601 UTC string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_meta_atomic(meta_file: Path, meta: Dict[str, Any]) -> None:
    """Atomically write task metadata via temp file + os.replace."""
    meta_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = json.dumps(meta, indent=2)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{meta_file.stem}.",
        suffix=".tmp",
        dir=str(meta_file.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
        retries = 5 if _IS_WINDOWS else 1
        for attempt in range(retries):
            try:
                os.replace(tmp_path, meta_file)
                break
            except PermissionError:
                if not _IS_WINDOWS or attempt == retries - 1:
                    raise
                time.sleep(0.01 * (attempt + 1))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


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
    """Parse exec/session JSONL lines in a single pass, updating both
    lifecycle analysis and live activity fields simultaneously.

    Codex 0.151+ emits retryable stream errors as top-level ``error`` events
    but strips the internal ``will_retry`` flag from public JSONL. Consequently
    only ``turn.completed`` and ``turn.failed`` are terminal. Error items are
    warnings. If both terminal event types occur, the stream is contradictory
    and the caller must prefer an observed process exit code.
    """
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
            state["thread_id"] = event.get("thread_id")
        elif etype == "session_meta":
            payload = event.get("payload", {})
            if isinstance(payload, dict) and not state["thread_id"]:
                state["thread_id"] = payload.get("id")

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
            # Non-terminal: this may be a retry notification whose will_retry
            # flag was removed by the public exec JSONL representation.
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
                            cmd_val[:150] if isinstance(cmd_val, str) else str(cmd_val)[:150]
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
                    # Config notices, runtime warnings and model reroutes use
                    # this form. They are explicitly non-terminal.
                    message = _error_message(item.get("message")) or _error_message(
                        item.get("error")
                    )
                    if message:
                        state["warnings"].append(message)
        elif etype == "response_item":
            # Session rollout format (fallback for callers that inspect one).
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                continue
            ptype = payload.get("type", "")
            if ptype == "function_call":
                state["tools_called"] += 1
                state["last_tool"] = payload.get("name", "?")
                raw_args = payload.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if isinstance(args, dict):
                        state["last_tool_args"] = (
                            args.get("cmd")
                            or args.get("command")
                            or args.get("path")
                            or args.get("pattern")
                            or str(args)[:150]
                        )
                    else:
                        state["last_tool_args"] = str(args)[:150]
                except (json.JSONDecodeError, TypeError):
                    state["last_tool_args"] = str(raw_args)[:150]
            elif ptype == "reasoning":
                content = payload.get("content") or []
                for c in content:
                    if isinstance(c, dict):
                        text_val = c.get("text", "")
                        if isinstance(text_val, str) and text_val:
                            state["last_reasoning"] = text_val[-200:]
            elif payload.get("role") == "assistant" and ptype == "message":
                content = payload.get("content") or []
                texts = [
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict)
                    and c.get("type") in ("output_text", "text", "input_text")
                ]
                if texts:
                    joined = "\n".join(texts)
                    state["last_assistant_text"] = joined
                    state["last_assistant_preview"] = texts[-1][-300:]
        elif etype == "event_msg":
            # Persisted rollouts use event_msg rather than public exec events.
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


def _finalize_jsonl_views(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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


def _parse_jsonl_full(jsonl_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Single-pass parse of JSONL text returning ``(analysis, status)``."""
    state = _new_jsonl_state()
    _consume_jsonl_lines(state, jsonl_text.splitlines())
    return _finalize_jsonl_views(state)


def _analyze_jsonl(jsonl_text: str) -> Dict[str, Any]:
    """Analyze exec/session JSONL without treating diagnostic errors as terminal."""
    analysis, _ = _parse_jsonl_full(jsonl_text)
    return analysis


# Incremental JSONL cache for large active/completed stdout files (>8 KiB).
# Small files are parsed directly in a single pass so in-place rewrites with
# coarse filesystem mtime granularity can never return stale data.
_JSONL_CACHE_MIN_BYTES = 8192
_JSONL_CACHE_MAX_ENTRIES = 128
_JSONL_CACHE: Dict[str, Dict[str, Any]] = {}
_jsonl_cache_lock = threading.Lock()


def _invalidate_jsonl_cache(stdout_path: Path) -> None:
    with _jsonl_cache_lock:
        _JSONL_CACHE.pop(str(stdout_path), None)


def _parse_jsonl_file(stdout_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse a task's ``.stdout`` JSONL file in a single pass, incrementally
    reading only newly appended bytes for large growing files.
    """
    try:
        st = stdout_path.stat()
    except OSError:
        _invalidate_jsonl_cache(stdout_path)
        return _finalize_jsonl_views(_new_jsonl_state())

    if st.st_size <= _JSONL_CACHE_MIN_BYTES:
        # Evict any prior large-file entry for the same path so truncating
        # and later regrowing the same inode can never splice stale state.
        _invalidate_jsonl_cache(stdout_path)
        try:
            text = stdout_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return _finalize_jsonl_views(_new_jsonl_state())
        return _parse_jsonl_full(text)

    key = str(stdout_path)
    with _jsonl_cache_lock:
        entry = _JSONL_CACHE.get(key)
        if (
            entry is not None
            and entry["ino"] == st.st_ino
            and entry["size"] == st.st_size
            and entry["mtime_ns"] == st.st_mtime_ns
        ):
            return _finalize_jsonl_views(entry["view_state"])

        if (
            entry is not None
            and entry["ino"] == st.st_ino
            and st.st_size > entry["size"]
            and entry["offset"] <= st.st_size
        ):
            try:
                with open(stdout_path, "rb") as f:
                    fst_before = os.fstat(f.fileno())
                    if (
                        fst_before.st_ino == entry["ino"]
                        and fst_before.st_size >= entry["size"]
                    ):
                        f.seek(entry["offset"])
                        chunk = f.read()
                        fst_after = os.fstat(f.fileno())
                        consumed_size = entry["offset"] + len(chunk)
                        if consumed_size >= entry["size"]:
                            last_nl = chunk.rfind(b"\n")
                            if last_nl >= 0:
                                complete = chunk[: last_nl + 1]
                                tail = chunk[last_nl + 1 :]
                                _consume_jsonl_lines(
                                    entry["state"],
                                    complete.decode("utf-8", errors="replace").splitlines(),
                                )
                                entry["offset"] += len(complete)
                            else:
                                tail = chunk

                            if tail.strip():
                                view_state = _clone_jsonl_state(entry["state"])
                                _consume_jsonl_lines(
                                    view_state, [tail.decode("utf-8", errors="replace")]
                                )
                            else:
                                view_state = entry["state"]

                            entry["size"] = consumed_size
                            entry["mtime_ns"] = fst_after.st_mtime_ns
                            entry["view_state"] = view_state
                            return _finalize_jsonl_views(view_state)
            except FileNotFoundError:
                _JSONL_CACHE.pop(key, None)
                return _finalize_jsonl_views(_new_jsonl_state())
            except OSError:
                return _finalize_jsonl_views(entry["view_state"])

        # Full single-pass read from byte 0 (cache miss, truncation, or new inode)
        try:
            with open(stdout_path, "rb") as f:
                raw = f.read()
                fst = os.fstat(f.fileno())
        except OSError:
            _JSONL_CACHE.pop(key, None)
            return _finalize_jsonl_views(_new_jsonl_state())

        state = _new_jsonl_state()
        last_nl = raw.rfind(b"\n")
        if last_nl >= 0:
            complete = raw[: last_nl + 1]
            tail = raw[last_nl + 1 :]
            _consume_jsonl_lines(
                state, complete.decode("utf-8", errors="replace").splitlines()
            )
            offset = len(complete)
        else:
            tail = raw
            offset = 0

        if tail.strip():
            view_state = _clone_jsonl_state(state)
            _consume_jsonl_lines(view_state, [tail.decode("utf-8", errors="replace")])
        else:
            view_state = state

        if len(raw) > _JSONL_CACHE_MIN_BYTES:
            if len(_JSONL_CACHE) >= _JSONL_CACHE_MAX_ENTRIES and key not in _JSONL_CACHE:
                oldest_key = next(iter(_JSONL_CACHE))
                _JSONL_CACHE.pop(oldest_key, None)
            _JSONL_CACHE[key] = {
                "ino": fst.st_ino,
                "size": len(raw),
                "mtime_ns": fst.st_mtime_ns,
                "offset": offset,
                "state": state,
                "view_state": view_state,
            }
        else:
            _JSONL_CACHE.pop(key, None)
        return _finalize_jsonl_views(view_state)


def _extract_thread_id_fast(
    stdout_path: Path, meta: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Extract ``thread_id`` from metadata or the opening lines of ``.stdout``
    without reading or parsing the entire output file.
    """
    if meta and meta.get("thread_id"):
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
                        if etype == "thread.started" and event.get("thread_id"):
                            return event["thread_id"]
                        if etype == "session_meta":
                            payload = event.get("payload", {})
                            if isinstance(payload, dict) and payload.get("id"):
                                return payload["id"]
                if bytes_read >= 65536:
                    break
    except OSError:
        return None
    return None


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


def _resolve_model(params: Optional[dict] = None) -> Optional[str]:
    """Resolve the effective model name from per-call params + SERVER_CONFIG."""
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
    """Extract the effective ``-m`` or ``-c model=`` value from task metadata.

    Stops scanning at ``--`` or the first positional argument so prompt text
    containing ``-m ...`` is never mistaken for a model flag.
    """
    if isinstance(command, (list, tuple)):
        tokens = [str(t) for t in command]
    elif isinstance(command, str) and command.strip():
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
    else:
        return None

    # Advance past launcher prefix (e.g. `codex exec [resume]` or `node codex.js exec [resume]`).
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
    model: Optional[str] = None,
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
        "model": model or _model_from_command(command),
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
    if analysis.get("saw_json") and analysis.get("last_assistant_text"):
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
    Single-pass with incremental caching for large files.
    Handles both `codex exec --json` format and session file format.
    """
    _, status = _parse_jsonl_file(stdout_path)
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

@contextlib.contextmanager
def _prompt_stdin(prompt: Optional[str], stdin_path: Optional[Path] = None):
    """Yield a readable file handle containing ``prompt`` as UTF-8, or DEVNULL.

    Passing prompts over stdin with ``-`` avoids CLI option injection for
    prompts starting with ``-``, avoids subcommand collisions (e.g. ``resume``),
    and removes the 32,767-char Windows command-line length cap on prompts.

    On POSIX, an unlinked ``TemporaryFile`` is used so the child reads via its
    inherited descriptor with no file left on disk. On Windows, ``TemporaryFile``
    uses ``O_TEMPORARY`` (delete-on-close), which is unsafe to rely on across
    ``node.exe`` -> ``codex.exe`` handle inheritance; instead a regular
    ``.stdin`` file is written and opened read-only:
      - When ``stdin_path`` is provided (async tasks), the file stays on disk
        for the child and is removed on launch failure, finalization/cancel, or
        old-task cleanup.
      - When ``stdin_path`` is omitted (sync runs), a unique ``sync_*.stdin``
        file is created in ``TASK_DIR`` and removed when the context exits
        (sync callers wrap both ``Popen`` and waiting for process exit).
    """
    if not prompt:
        yield subprocess.DEVNULL
        return

    payload = prompt.encode("utf-8")
    if not _IS_WINDOWS:
        with tempfile.TemporaryFile(mode="w+b") as tmp:
            tmp.write(payload)
            tmp.flush()
            tmp.seek(0)
            yield tmp
        return

    # Windows: write a regular .stdin file and hand the child a normal read-only
    # handle (no O_TEMPORARY).
    cleanup_on_exit = stdin_path is None
    if stdin_path is None:
        TASK_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp_name = tempfile.mkstemp(
            prefix="sync_", suffix=".stdin", dir=str(TASK_DIR)
        )
        path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as wf:
                fd = -1
                wf.write(payload)
        except Exception:
            if fd >= 0:
                os.close(fd)
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            raise
    else:
        path = stdin_path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(
            str(path),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            with os.fdopen(fd, "wb") as wf:
                fd = -1
                wf.write(payload)
        finally:
            if fd >= 0:
                os.close(fd)

    try:
        with open(path, "rb") as fh:
            yield fh
    finally:
        if cleanup_on_exit:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)


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

    Prompts are represented as ``-`` and supplied via ``_prompt_stdin(prompt)``
    so prompts starting with ``-`` cannot be parsed as options and long prompts
    do not hit OS argv limits.

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
    if isinstance(per_call, dict):
        for k, v in per_call.items():
            _flatten_config(str(k), v, merged)

    # Model
    model = params.get("model")
    if model:
        cmd.extend(["-m", model])
        merged.pop("model", None)
    elif "model" in merged:
        raw_model = _strip_toml_quotes(str(merged.pop("model")))
        if raw_model:
            cmd.extend(["-m", raw_model])

    # Sandbox
    sandbox = params.get("sandbox")
    if sandbox:
        merged["sandbox_mode"] = sandbox
    if not is_resume and "sandbox_mode" in merged:
        raw_sandbox = _strip_toml_quotes(str(merged.pop("sandbox_mode")))
        if raw_sandbox:
            cmd.extend(["-s", raw_sandbox])

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
        tid_str = str(thread_id)
        if tid_str.startswith("-"):
            cmd.append("--")
        cmd.append(tid_str)

    prompt = params.get("prompt", "")
    if prompt:
        cmd.append("-")

    # Fresh exec passes -C, so a relative cwd resolves inside codex against
    # the server's cwd; resume has no -C, so resolve it here instead.
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
    """Build a `codex exec resume` command for continuing a session."""
    params: Dict[str, Any] = dict(extra_params or {})
    params["threadId"] = thread_id
    params["prompt"] = prompt
    cmd, _ = _build_command(params)
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
_PROCESS_CHECK_CMD = (
    "`tasklist /FI \"IMAGENAME eq codex.exe\"`"
    if _IS_WINDOWS
    else "`pgrep -af 'codex exec'`"
)

_UNVERIFIED_FAILURE_HINT = (
    "NOTE: this verdict is inferred, not observed -- the exit code was lost to "
    "a reaping race, so the status was derived from output alone. Codex may "
    "have finished its work. Verify before reporting failure upstream: run "
    f"{_PROCESS_CHECK_CMD} to see whether it is still alive, then check "
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
    f"{_PROCESS_CHECK_CMD}."
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
    if not _is_valid_request_id(request_id):
        return False
    with _sync_procs_lock:
        proc = _SYNC_PROCS.get(request_id)
    if proc is None or proc.poll() is not None:
        return False
    _stop_proc(proc, own_group=True)
    logging.info("Terminated sync child pid=%s for request id=%s", proc.pid, request_id)
    return True


def _terminate_all_sync_procs() -> None:
    """Kill any surviving sync child process trees on server shutdown."""
    with _sync_procs_lock:
        items = list(_SYNC_PROCS.items())
    for request_id, proc in items:
        alive = proc.poll() is None
        group_alive = (not _IS_WINDOWS) and _posix_group_alive(proc.pid)
        if alive or group_alive:
            logging.info("Shutdown: killing sync child pid=%s (request %s)", proc.pid, request_id)
            if _IS_WINDOWS:
                _win_kill_tree(proc.pid)
            else:
                _posix_signal_tree(proc.pid, signal.SIGKILL, own_group=True)
            if alive:
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass


def _send_progress(
    progress_token: Any,
    elapsed: float,
    label: str,
    detail: Optional[str] = None,
) -> None:
    """
    Emit an MCP progress notification.

    This is what keeps a long sync run alive: the client's idle timer measures
    silence, not duration, so a run with no wire traffic is indistinguishable
    from a hung server no matter how much real work it is doing.
    """
    if progress_token is None:
        return
    message = f"{label} running ({int(elapsed)}s elapsed)"
    if detail:
        message = f"{message} -- {detail}"
    _send({
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {
            "progressToken": progress_token,
            "progress": round(elapsed, 1),
            "message": message,
        },
    })


def _format_task_activity_brief(jsonl_status: Dict[str, Any]) -> Optional[str]:
    """Return a short one-line activity summary from ``_parse_jsonl_status``,
    or None if no meaningful activity has been observed yet.
    """
    tools_called = jsonl_status.get("tools_called", 0)
    last_tool = jsonl_status.get("last_tool")
    last_tool_args = jsonl_status.get("last_tool_args")
    phase = jsonl_status.get("phase", "starting")

    if last_tool:
        tool_str = last_tool
        if last_tool_args:
            tool_str += f"({last_tool_args[:60]})"
        noun = "tool" if tools_called == 1 else "tools"
        return f"{tools_called} {noun} called, last: {tool_str}"
    if tools_called:
        noun = "tool" if tools_called == 1 else "tools"
        return f"{tools_called} {noun} called"
    if jsonl_status.get("last_assistant_text"):
        return jsonl_status["last_assistant_text"][:100].replace("\n", " ")
    if jsonl_status.get("last_reasoning"):
        return jsonl_status["last_reasoning"][:100].replace("\n", " ")
    if phase and phase != "starting":
        return phase
    return None


def _summarize_pending_activity(pending: List[str]) -> Optional[str]:
    """Build an enriched progress detail string across pending async tasks."""
    if not pending:
        return None
    if len(pending) == 1:
        st = _parse_jsonl_status(TASK_DIR / f"{pending[0]}.stdout")
        return _format_task_activity_brief(st)

    snippets = []
    for tid in pending[:3]:
        st = _parse_jsonl_status(TASK_DIR / f"{tid}.stdout")
        brief = _format_task_activity_brief(st)
        if brief:
            snippets.append(f"{tid[:8]}: {brief}")
    if not snippets:
        return None
    if len(pending) > 3:
        snippets.append(f"+{len(pending) - 3} more")
    return "; ".join(snippets)


def _wait_proc_detailed(
    proc: subprocess.Popen,
    deadline: Optional[float] = None,
    request_id: Any = None,
    timeout_msg: str = "Error: Codex execution timed out",
    progress_token: Any = None,
    progress_label: str = "Codex",
    command: Any = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Wait for a Popen process with cancellation, timeout and progress support.
    Returns a detailed outcome dict shared by sync codex and reply paths.
    """
    started = time.time()
    last_progress = started
    while True:
        remaining = None
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                _stop_proc(proc)
                return {
                    "text": timeout_msg,
                    "thread_id": None,
                    "failed": True,
                    "status": "failed",
                    "exit_code": proc.returncode,
                    "model": model or _model_from_command(command),
                    "failure": None,
                    "partial_output": None,
                }
        wait_time = min(_POLL_INTERVAL, remaining) if remaining is not None else _POLL_INTERVAL
        try:
            stdout, stderr = proc.communicate(timeout=wait_time)
            analysis = _analyze_jsonl(stdout)
            status, _ = _decide_run_status(proc.returncode, analysis, stderr)
            if status == "failed":
                failure = _build_failure_info(
                    stdout, stderr, command=command, analysis=analysis, model=model
                )
                exit_code = proc.returncode if proc.returncode is not None else "?"
                text = (
                    f"{progress_label} FAILED (exit {exit_code})\n"
                    f"{_format_failure_info(failure)}"
                )
                return {
                    "text": text,
                    "thread_id": analysis.get("thread_id"),
                    "failed": True,
                    "status": "failed",
                    "exit_code": proc.returncode,
                    "model": failure.get("model"),
                    "failure": failure,
                    "partial_output": analysis.get("last_assistant_text"),
                }
            result, thread_id = _extract_result(stdout, stderr, analysis=analysis)
            return {
                "text": result,
                "thread_id": thread_id,
                "failed": False,
                "status": "completed",
                "exit_code": proc.returncode,
                "model": model or _model_from_command(command),
                "failure": None,
                "partial_output": None,
            }
        except subprocess.TimeoutExpired:
            if request_id is not None and _is_cancelled(request_id):
                _stop_proc(proc)
                return {
                    "text": "Cancelled by client",
                    "thread_id": None,
                    "failed": True,
                    "status": "cancelled",
                    "exit_code": proc.returncode,
                    "model": model or _model_from_command(command),
                    "failure": None,
                    "partial_output": None,
                }
            now = time.time()
            if now - last_progress >= _PROGRESS_INTERVAL:
                last_progress = now
                _send_progress(progress_token, now - started, progress_label)


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
    """Compatibility wrapper over ``_wait_proc_detailed`` returning ``(text, thread_id, failed)``."""
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
    """
    Run codex synchronously with cancellation support and structured outcome.

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
    resolved_model = _resolve_model(cmd_params)
    proc_started = False

    worktree_enabled = bool(cmd_params.pop("worktree", False))

    try:
        if worktree_enabled:
            _cleanup_old_tasks()
            run_id = secrets.token_hex(4)
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
        logging.info("Sync exec: %s", shlex.join(cmd))
        with _prompt_stdin(cmd_params.get("prompt", "")) as stdin_fh:
            proc = subprocess.Popen(
                _exec_argv(cmd),
                stdin=stdin_fh,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                **_TEXT_KW,
                **_popen_kw(new_group=True),
            )
            if stdin_fh is not subprocess.DEVNULL:
                stdin_fh.close()
            proc_started = True
            _register_sync_proc(request_id, proc)
            outcome = _wait_proc_detailed(
                proc,
                request_id=request_id,
                timeout_msg="Error: Codex execution timed out",
                progress_token=progress_token,
                progress_label="Codex",
                command=cmd,
                model=resolved_model,
            )
        thread_id = outcome["thread_id"]
        exit_code = outcome["exit_code"]
        meta_status = outcome["status"]
        outcome["worktree_info"] = worktree_info
        return outcome
    except Exception as exc:
        if worktree_info and not proc_started:
            _remove_worktree(
                worktree_info["worktree_root"], worktree_info["worktree_branch"]
            )
            worktree_info = None
        return {
            "text": f"Error calling Codex: {exc}",
            "thread_id": None,
            "worktree_info": worktree_info,
            "failed": True,
            "status": "failed",
            "exit_code": None,
            "model": resolved_model,
            "failure": None,
            "partial_output": None,
        }
    finally:
        _unregister_sync_proc(request_id)
        if worktree_info and run_id:
            completed_at = time.time()
            meta: Dict[str, Any] = {
                "task_id": run_id,
                "status": meta_status,
                "started_at": started_at,
                "completed_at": completed_at,
                "last_updated_at": completed_at,
                "worktree_root": worktree_info["worktree_root"],
                "worktree_path": worktree_info["worktree_path"],
                "worktree_branch": worktree_info["worktree_branch"],
            }
            if cmd:
                meta["command"] = shlex.join(cmd)
            if resolved_model:
                meta["model"] = resolved_model
            if exit_code is not None:
                meta["exit_code"] = exit_code
            if thread_id:
                meta["thread_id"] = thread_id
            try:
                _write_meta_atomic(TASK_DIR / f"{run_id}.meta", meta)
            except Exception as exc:
                logging.warning("Failed to persist sync worktree metadata for %s: %s", run_id, exc)


def _run_sync(
    params: dict,
    request_id: Any = None,
    progress_token: Any = None,
) -> Tuple[str, Optional[str], Optional[Dict[str, str]]]:
    """Compatibility wrapper returning ``(text, thread_id, worktree_info)``."""
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
    """
    Run codex reply synchronously with cancellation support and structured outcome.

    Deliberately has no timeout: sync tools run to completion, and callers that
    need a bounded wait should use codex_async(threadId=...) + codex_wait.
    Models set a low timeout here whatever the description says, which turned
    long resumes into spurious failures.
    """
    params: Dict[str, Any] = dict(extra_params or {})
    params["threadId"] = thread_id
    params["prompt"] = prompt
    resolved_model = _resolve_model(params)
    cmd, cwd = _build_command(params)
    cmd = _with_json_flag(cmd)
    logging.info("Reply exec: %s", shlex.join(cmd))
    proc = None
    try:
        popen_kwargs: Dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            **_TEXT_KW,
            **_popen_kw(new_group=True),
        }
        if cwd is not None:
            popen_kwargs["cwd"] = cwd
        with _prompt_stdin(prompt) as stdin_fh:
            proc = subprocess.Popen(
                _exec_argv(cmd),
                stdin=stdin_fh,
                **popen_kwargs,
            )
            if stdin_fh is not subprocess.DEVNULL:
                stdin_fh.close()
            _register_sync_proc(request_id, proc)
            outcome = _wait_proc_detailed(
                proc,
                request_id=request_id,
                timeout_msg="Error: Codex reply timed out",
                progress_token=progress_token,
                progress_label="Codex reply",
                command=cmd,
                model=resolved_model,
            )
        if not outcome["thread_id"]:
            outcome["thread_id"] = thread_id
        return outcome
    except Exception as exc:
        return {
            "text": f"Error calling Codex reply: {exc}",
            "thread_id": thread_id,
            "failed": True,
            "status": "failed",
            "exit_code": None,
            "model": resolved_model,
            "failure": None,
            "partial_output": None,
        }
    finally:
        if proc is not None:
            _unregister_sync_proc(request_id)


def _run_reply_sync(
    thread_id: str,
    prompt: str,
    request_id: Any = None,
    progress_token: Any = None,
) -> Tuple[str, Optional[str]]:
    """Compatibility wrapper returning ``(text, thread_id)``."""
    outcome = _run_reply_sync_detailed(
        thread_id, prompt, request_id=request_id, progress_token=progress_token
    )
    return outcome["text"], outcome["thread_id"]

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
            task_id = meta_file.stem
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            status = meta.get("status")

            # Finalize unpolled "running" tasks whose process has died
            if status == "running":
                pid = meta.get("pid")
                pid_start_time = meta.get("pid_start_time")
                if _is_alive(pid, expected_start_time=pid_start_time):
                    continue  # still running, skip
                # Dead but never finalized -- resolve it now (acquires task lock)
                _resolve_task_state(task_id)

            lock = _get_task_lock(task_id)
            with lock:
                if not meta_file.exists():
                    continue
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
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
                _invalidate_jsonl_cache(TASK_DIR / f"{task_id}.stdout")
                for ext in (".meta", ".stdout", ".stderr", ".stdin"):
                    try:
                        (TASK_DIR / f"{task_id}{ext}").unlink(missing_ok=True)
                    except OSError:
                        pass

            # Prune per-task finalization lock
            with _task_finalize_guard:
                _task_finalize_locks.pop(task_id, None)

            logging.debug("Cleaned up old task %s", task_id)
        except Exception:
            continue

    # Clean up any orphaned Windows sync prompt files left by a hard crash
    for stdin_file in TASK_DIR.glob("sync_*.stdin"):
        try:
            if now - stdin_file.stat().st_mtime >= _TASK_MAX_AGE:
                stdin_file.unlink(missing_ok=True)
        except OSError:
            continue


def _start_async_internal(
    params: dict,
    *,
    high_entropy_id: bool = False,
    mcp_task: bool = False,
    tool_name: str = "codex_async",
) -> Dict[str, Any]:
    _cleanup_old_tasks()

    task_id = secrets.token_hex(16) if high_entropy_id else secrets.token_hex(4)
    cmd_params = dict(params)
    worktree_enabled = bool(cmd_params.pop("worktree", False))
    worktree_root: Optional[str] = None
    worktree_path: Optional[str] = None
    worktree_branch: Optional[str] = None
    resolved_model = _resolve_model(cmd_params)

    if worktree_enabled:
        worktree_root, wt_cwd, worktree_branch = _create_worktree(task_id, cmd_params.get("cwd"))
        worktree_path = wt_cwd
        cmd_params["cwd"] = wt_cwd

    stdout_f: Optional[Path] = None
    stderr_f: Optional[Path] = None
    stdin_f: Optional[Path] = None
    try:
        cmd, cwd = _build_command(cmd_params)
        # Add --json for structured output (enables live status parsing)
        cmd = _with_json_flag(cmd)
        logging.info("Async start [%s]: %s", task_id, shlex.join(cmd))

        TASK_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        stdout_f = TASK_DIR / f"{task_id}.stdout"
        stderr_f = TASK_DIR / f"{task_id}.stderr"
        stdin_f = (TASK_DIR / f"{task_id}.stdin") if _IS_WINDOWS else None

        with open(stdout_f, "w", encoding="utf-8") as out, open(
            stderr_f, "w", encoding="utf-8"
        ) as err:
            with _prompt_stdin(
                cmd_params.get("prompt", ""), stdin_path=stdin_f
            ) as stdin_fh:
                proc = subprocess.Popen(
                    _exec_argv(cmd),
                    stdin=stdin_fh,
                    stdout=out,
                    stderr=err,
                    cwd=cwd,
                    **_TEXT_KW,
                    **_popen_kw(detach=True),
                )
    except Exception:
        # Clean up worktree and empty log/stdin files if task launch fails
        for path in (stdout_f, stderr_f, stdin_f):
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        if worktree_root and worktree_branch:
            _remove_worktree(worktree_root, worktree_branch)
        raise

    # Track PID for SIGCHLD handler (sentinel replaced on reap)
    _ASYNC_PIDS[proc.pid] = _NOT_REAPED
    # Keep Popen alive so Python's finalizer doesn't steal the exit code
    _ASYNC_PROCS[proc.pid] = proc

    started_at = time.time()
    meta: Dict[str, Any] = {
        "task_id": task_id,
        "pid": proc.pid,
        "pid_start_time": _get_pid_start_time(proc.pid),
        "status": "running",
        "command": shlex.join(cmd),
        "started_at": started_at,
        "last_updated_at": started_at,
    }
    if not _IS_WINDOWS:
        meta["own_process_group"] = True
    if resolved_model:
        meta["model"] = resolved_model
    if mcp_task:
        meta["mcp_task"] = True
        meta["tool_name"] = tool_name
    if cmd_params.get("threadId"):
        meta["thread_id"] = str(cmd_params["threadId"])
    if worktree_root:
        meta["worktree_root"] = worktree_root
    if worktree_path:
        meta["worktree_path"] = worktree_path
    if worktree_branch:
        meta["worktree_branch"] = worktree_branch
    _write_meta_atomic(TASK_DIR / f"{task_id}.meta", meta)

    return {
        "task_id": task_id,
        "worktree_path": worktree_path,
        "worktree_branch": worktree_branch,
        "model": resolved_model,
        "started_at": started_at,
    }


def _start_async(params: dict) -> Dict[str, Optional[str]]:
    """Launch an async Codex task and return its 8-char task_id and worktree info."""
    return _start_async_internal(params)


# ===================================================================
# Centralized task state resolution
# ===================================================================

def _terminal_state_dict(task_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    started_at = meta.get("started_at", time.time())
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


def _finalize_dead_task_locked(
    task_id: str, meta_file: Path, meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Finalize a dead task while holding ``_get_task_lock(task_id)``."""
    started_at = meta.get("started_at", time.time())
    pid = meta.get("pid")

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
        if exit_code is None and pid is not None and not _IS_WINDOWS:
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
    stdout_path = TASK_DIR / f"{task_id}.stdout"
    stderr_text = _safe_read(TASK_DIR / f"{task_id}.stderr")
    analysis, _ = _parse_jsonl_file(stdout_path)
    final_status, status_inferred = _decide_run_status(
        exit_code, analysis, stderr_text
    )

    # Persist final state to metadata atomically
    meta["status"] = final_status
    meta["completed_at"] = completed_at
    meta["last_updated_at"] = completed_at
    if exit_code is not None:
        meta["exit_code"] = exit_code
    if exit_code_lost:
        meta["exit_code_lost"] = True
    meta["status_inferred"] = status_inferred
    if analysis.get("thread_id") and not meta.get("thread_id"):
        meta["thread_id"] = analysis["thread_id"]
    try:
        _write_meta_atomic(meta_file, meta)
    except Exception as exc:
        logging.warning("Failed to persist final metadata for %s: %s", task_id, exc)

    # Cleanup in-memory tracking and any Windows .stdin prompt file
    if pid is not None:
        _ASYNC_PIDS.pop(pid, None)
        _ASYNC_PROCS.pop(pid, None)
    try:
        (TASK_DIR / f"{task_id}.stdin").unlink(missing_ok=True)
    except OSError:
        pass

    return {
        "status": final_status,
        "task_id": task_id,
        "meta": meta,
        "elapsed_seconds": int(completed_at - started_at),
        "exit_code": exit_code,
        "exit_code_lost": exit_code_lost,
        "status_inferred": status_inferred,
    }


def _resolve_task_state(task_id: str) -> Dict[str, Any]:
    """
    Single source of truth for task lifecycle state.

    Returns dict with:
      - status: "running" | "completed" | "failed" | "cancelled" | "not_found" | "error"
      - task_id
      - meta (if status is running/completed/failed/cancelled)
      - elapsed_seconds (if status is running/completed/failed/cancelled)
      - exit_code (if completed/failed, may be None)
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
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "error", "task_id": task_id, "error": f"Bad metadata: {exc}"}

    started_at = meta.get("started_at", time.time())

    # Already finalized in a previous call
    if meta.get("status") in ("completed", "failed", "cancelled"):
        return _terminal_state_dict(task_id, meta)

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
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"status": "error", "task_id": task_id, "error": f"Bad metadata: {exc}"}
        if meta.get("status") in ("completed", "failed", "cancelled"):
            return _terminal_state_dict(task_id, meta)

        return _finalize_dead_task_locked(task_id, meta_file, meta)


def _check_task(
    task_id: str, state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Check task status and return formatted result dict."""
    state = state if state is not None else _resolve_task_state(task_id)
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
        if meta.get("model"):
            resp["model"] = meta["model"]
        if meta.get("worktree_path"):
            resp["worktree_path"] = meta["worktree_path"]
        if meta.get("worktree_branch"):
            resp["worktree_branch"] = meta["worktree_branch"]
        tid = _extract_thread_id_fast(TASK_DIR / f"{task_id}.stdout", meta)
        if tid:
            resp["thread_id"] = tid
        return resp

    # completed, failed, or cancelled
    stdout_path = TASK_DIR / f"{task_id}.stdout"
    stderr_path = TASK_DIR / f"{task_id}.stderr"
    analysis, _ = _parse_jsonl_file(stdout_path)
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    if analysis.get("saw_json") and analysis.get("last_assistant_text"):
        result = analysis["last_assistant_text"]
        thread_id = analysis.get("thread_id")
    else:
        stdout = _safe_read(stdout_path)
        stderr = _safe_read(stderr_path)
        result, thread_id = _extract_result(stdout, stderr, analysis=analysis)
    meta = state.get("meta", {})
    thread_id = thread_id or meta.get("thread_id")

    resp = {
        "status": status,
        "task_id": task_id,
        "result": result,
        "elapsed_seconds": state["elapsed_seconds"],
    }
    if meta.get("model"):
        resp["model"] = meta["model"]
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
        if stderr is None:
            stderr = _safe_read(stderr_path)
        if stdout is None:
            # Only read raw stdout when no structured JSONL error cause exists
            stdout = (
                ""
                if (analysis.get("terminal_error") or analysis.get("errors"))
                else _safe_read(stdout_path)
            )
        failure = _build_failure_info(
            stdout,
            stderr,
            command=meta.get("command"),
            analysis=analysis,
            model=meta.get("model"),
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
    timeout: Optional[float] = None,
    request_id: Any = None,
    progress_token: Any = None,
) -> Dict[str, Any]:
    """
    Block until all tasks complete, timeout, or request is cancelled.

    Polls only liveness via ``_resolve_task_state`` while tasks are running
    (avoiding repeated full ``.stdout`` reads every 2s), and enriches periodic
    progress heartbeats with live JSONL activity.
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
                        "elapsed_seconds": state.get("elapsed_seconds", 0)
                        if isinstance(state, dict)
                        else 0,
                        "error": "Wait cancelled by client",
                        "wait_cancelled": True,
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
            state = _resolve_task_state(tid)
            if state["status"] in ("completed", "failed", "cancelled", "not_found", "error"):
                results[tid] = _check_task(tid, state=state)
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
                    "elapsed_seconds": state.get("elapsed_seconds", 0)
                    if isinstance(state, dict)
                    else 0,
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
            detail = (
                _summarize_pending_activity(pending)
                if progress_token is not None
                else None
            )
            _send_progress(
                progress_token,
                now - started,
                f"{waiting_on} Codex task{'s' if waiting_on != 1 else ''}",
                detail=detail,
            )

        sleep_time = 2.0
        if deadline is not None:
            sleep_time = min(sleep_time, max(0.1, deadline - time.time()))
        time.sleep(sleep_time)

    return results

# ===================================================================
# Server instructions & Tool definitions
# ===================================================================

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

_FAILURE_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {"type": "string"},
        "cause": {"type": "string"},
        "category": {"type": "string"},
        "retryable": {"type": "boolean"},
        "action": {"type": "string"},
        "stdout_diagnostics": {"type": "string"},
        "stderr_diagnostics": {"type": "string"},
    },
    "required": ["cause", "category"],
}

_CODEX_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["completed", "failed", "cancelled"]},
        "text": {"type": "string"},
        "thread_id": {"type": "string"},
        "model": {"type": "string"},
        "exit_code": {"type": "integer"},
        "worktree_path": {"type": "string"},
        "worktree_branch": {"type": "string"},
        "partial_output": {"type": "string"},
        "failure": _FAILURE_SCHEMA,
    },
    "required": ["status", "text"],
}

_CODEX_ASYNC_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["started", "error"]},
        "task_id": {"type": "string"},
        "model": {"type": "string"},
        "worktree_path": {"type": "string"},
        "worktree_branch": {"type": "string"},
        "error": {"type": "string"},
    },
    "required": ["status"],
}

_CODEX_REPLY_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["completed", "failed", "cancelled"]},
        "text": {"type": "string"},
        "thread_id": {"type": "string"},
        "model": {"type": "string"},
        "exit_code": {"type": "integer"},
        "partial_output": {"type": "string"},
        "failure": _FAILURE_SCHEMA,
    },
    "required": ["status", "text"],
}

_CODEX_STATUS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "status": {"type": "string"},
                    "elapsed_seconds": {"type": "integer"},
                    "phase": {"type": "string"},
                    "tools_called": {"type": "integer"},
                    "thread_id": {"type": "string"},
                    "model": {"type": "string"},
                    "worktree_path": {"type": "string"},
                    "worktree_branch": {"type": "string"},
                    "last_tool": {"type": "string"},
                    "last_tool_args": {"type": "string"},
                    "last_reasoning": {"type": "string"},
                    "last_assistant_text": {"type": "string"},
                    "last_warning": {"type": "string"},
                    "last_error": {"type": "string"},
                    "exit_code": {"type": "integer"},
                    "exit_code_lost": {"type": "boolean"},
                    "status_inferred": {"type": "boolean"},
                    "terminal_conflict": {"type": "boolean"},
                    "failure": _FAILURE_SCHEMA,
                    "error": {"type": "string"},
                },
                "required": ["task_id", "status"],
            },
        },
    },
    "required": ["tasks"],
}

_CODEX_WAIT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "status": {"type": "string"},
                    "elapsed_seconds": {"type": "integer"},
                    "result": {"type": "string"},
                    "thread_id": {"type": "string"},
                    "model": {"type": "string"},
                    "worktree_path": {"type": "string"},
                    "worktree_branch": {"type": "string"},
                    "exit_code": {"type": "integer"},
                    "exit_code_lost": {"type": "boolean"},
                    "status_inferred": {"type": "boolean"},
                    "failure_cause": {"type": "string"},
                    "failure_category": {"type": "string"},
                    "retryable": {"type": "boolean"},
                    "failure_action": {"type": "string"},
                    "partial_output": {"type": "string"},
                    "stdout_diagnostics": {"type": "string"},
                    "stderr_diagnostics": {"type": "string"},
                    "error": {"type": "string"},
                },
                "required": ["task_id", "status"],
            },
        },
    },
    "required": ["tasks"],
}

_CODEX_CANCEL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string"},
        "status": {"type": "string"},
        "message": {"type": "string"},
        "error": {"type": "string"},
        "worktree_path": {"type": "string"},
        "worktree_branch": {"type": "string"},
        "exit_code": {"type": "integer"},
        "exit_code_lost": {"type": "boolean"},
        "status_inferred": {"type": "boolean"},
        "model": {"type": "string"},
        "failure_cause": {"type": "string"},
        "failure_category": {"type": "string"},
        "retryable": {"type": "boolean"},
        "failure_action": {"type": "string"},
    },
    "required": ["task_id", "status"],
}

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
            "supported_protocol_versions": list(_ALL_PROTOCOL_VERSIONS),
            "extensions": [_TASKS_EXTENSION_ID],
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
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                task_id = meta.get("task_id", meta_file.stem)
                state = _resolve_task_state(task_id)
                resolved_meta = state.get("meta") if isinstance(state.get("meta"), dict) else meta
                entry: Dict[str, Any] = {
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
                for key in ("model", "worktree_path", "worktree_branch", "thread_id"):
                    val = resolved_meta.get(key)
                    if val:
                        entry[key] = val
                if state.get("status") == "failed":
                    stdout_path = TASK_DIR / f"{task_id}.stdout"
                    analysis, _ = _parse_jsonl_file(stdout_path)
                    stdout = (
                        ""
                        if (analysis.get("terminal_error") or analysis.get("errors"))
                        else _safe_read(stdout_path)
                    )
                    stderr = _safe_read(TASK_DIR / f"{task_id}.stderr")
                    failure = _build_failure_info(
                        stdout,
                        stderr,
                        command=resolved_meta.get("command"),
                        analysis=analysis,
                        model=resolved_meta.get("model"),
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


def _build_already_terminal_cancel_resp(
    task_id: str, state: Dict[str, Any]
) -> Dict[str, Any]:
    status = state["status"]
    resp: Dict[str, Any] = {
        "status": status,
        "task_id": task_id,
        "message": f"Task already {status}",
        "already_terminal": True,
    }
    meta = state.get("meta", {})
    if meta.get("worktree_path"):
        resp["worktree_path"] = meta["worktree_path"]
    if meta.get("worktree_branch"):
        resp["worktree_branch"] = meta["worktree_branch"]
    if status == "failed":
        failure = _check_task(task_id, state=state)
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


def _cancel_task(task_id: str) -> Dict[str, Any]:
    """Kill a running async task (and its process group) and mark it as cancelled."""
    if not _validate_task_id(task_id):
        return {"status": "error", "task_id": task_id, "error": f"Invalid task ID: {task_id}"}

    meta_file = TASK_DIR / f"{task_id}.meta"
    if not meta_file.exists():
        return {"status": "not_found", "task_id": task_id, "error": f"Task {task_id} not found"}

    lock = _get_task_lock(task_id)
    with lock:
        if not meta_file.exists():
            return {"status": "not_found", "task_id": task_id, "error": f"Task {task_id} not found"}

        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"status": "error", "task_id": task_id, "error": f"Bad metadata: {exc}"}

        if meta.get("status") in ("completed", "failed", "cancelled"):
            terminal_state = _terminal_state_dict(task_id, meta)
            return _build_already_terminal_cancel_resp(task_id, terminal_state)

        pid = meta.get("pid")
        pid_start_time = meta.get("pid_start_time")

        if not (pid and _is_alive(pid, expected_start_time=pid_start_time)):
            # Process already died before cancellation -- preserve its real
            # completed/failed verdict instead of overwriting as cancelled.
            final_state = _finalize_dead_task_locked(task_id, meta_file, meta)
            return _build_already_terminal_cancel_resp(task_id, final_state)

        if _IS_WINDOWS:
            # No SIGTERM/SIGKILL on Windows, and os.kill would only reach the node
            # launcher; take down the whole tree instead.
            if not _win_kill_tree(pid):
                # Leave it running/tracked so a retry is possible; marking it
                # cancelled would hide a process that is still editing files.
                return {
                    "status": "error",
                    "task_id": task_id,
                    "error": f"Failed to kill process tree of PID {pid}; task is still running",
                }
            logging.info("Killed process tree of PID %d (task %s)", pid, task_id)
        else:
            own_group = bool(meta.get("own_process_group"))
            _posix_signal_tree(pid, signal.SIGTERM, own_group=own_group)
            logging.info("Sent SIGTERM to PID %d (group=%s, task %s)", pid, own_group, task_id)

            # Poll briefly for clean exit before escalating to SIGKILL
            for _ in range(10):
                time.sleep(0.05)
                if not _is_alive(pid, expected_start_time=pid_start_time):
                    break

            if _is_alive(pid, expected_start_time=pid_start_time):
                _posix_signal_tree(pid, signal.SIGKILL, own_group=own_group)
                logging.info("Sent SIGKILL to PID %d (group=%s, task %s)", pid, own_group, task_id)
            elif own_group and _posix_group_alive(pid):
                # Leader exited on SIGTERM, but child processes remain in its group
                time.sleep(0.05)
                if _posix_group_alive(pid):
                    _posix_signal_tree(pid, signal.SIGKILL, own_group=True)
                    logging.info("Sent SIGKILL to surviving process group %d (task %s)", pid, task_id)

        # Reap the terminated child before dropping tracking so SIGKILLed
        # processes cannot linger as zombies or trigger Popen.__del__ warnings.
        proc = _ASYNC_PROCS.get(pid) if pid else None
        if proc is not None:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        elif pid and not _IS_WINDOWS:
            try:
                os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                pass

        # Mark as cancelled in metadata atomically
        now = time.time()
        meta["status"] = "cancelled"
        meta["completed_at"] = now
        meta["last_updated_at"] = now
        try:
            _write_meta_atomic(meta_file, meta)
        except Exception as exc:
            logging.warning("Failed to persist cancelled metadata for %s: %s", task_id, exc)

        # Cleanup in-memory tracking and any Windows .stdin prompt file
        if pid:
            _ASYNC_PIDS.pop(pid, None)
            _ASYNC_PROCS.pop(pid, None)
        try:
            (TASK_DIR / f"{task_id}.stdin").unlink(missing_ok=True)
        except OSError:
            pass

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
# Structured content & MCP Tasks (io.modelcontextprotocol/tasks) helpers
# ===================================================================

def _clean_none(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy omitting keys whose values are None."""
    return {k: v for k, v in mapping.items() if v is not None}


def _failure_structured(failure: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not failure:
        return None
    return _clean_none({
        "model": failure.get("model"),
        "cause": failure.get("cause"),
        "category": failure.get("category"),
        "retryable": failure.get("retryable"),
        "action": failure.get("action"),
        "stdout_diagnostics": failure.get("stdout_diagnostics"),
        "stderr_diagnostics": failure.get("stderr_diagnostics"),
    })


def _client_supports_tasks(
    client_caps: Dict[str, Any], req_meta: Dict[str, Any]
) -> bool:
    extensions = client_caps.get("extensions")
    if isinstance(extensions, dict) and _TASKS_EXTENSION_ID in extensions:
        val = extensions[_TASKS_EXTENSION_ID]
        return val is not False and val is not None
    if _TASKS_EXTENSION_ID in req_meta:
        val = req_meta[_TASKS_EXTENSION_ID]
        return val is not False and val is not None
    return False


def _format_mcp_task_status_message(
    jsonl_status: Dict[str, Any], elapsed: int
) -> str:
    """Build a human-readable ``statusMessage`` for ``tasks/get`` while working."""
    brief = _format_task_activity_brief(jsonl_status)
    if brief:
        msg = f"Running ({elapsed}s): {brief}"
    else:
        msg = f"Starting Codex session ({elapsed}s elapsed)"
    if jsonl_status.get("last_error"):
        msg += f" [non-terminal error: {jsonl_status['last_error'][:120]}]"
    return msg


def _build_mcp_task_tool_result(
    info: Dict[str, Any], meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the terminal ``CallToolResult`` embedded in a completed ``tasks/get``
    (including tool-level Codex failures with ``isError: true`` per SEP-2663).
    """
    status = info["status"]
    tool_name = meta.get("tool_name", "codex")
    is_reply = tool_name == "codex_reply"
    label = "Codex reply" if is_reply else "Codex"

    if status == "completed":
        text = info["result"]
        if info.get("status_inferred"):
            text = f"{_UNVERIFIED_COMPLETION_HINT}\n\n{text}"
        details = []
        if info.get("thread_id") and not is_reply:
            details.append(
                f"Thread ID: {info['thread_id']}\n"
                f'Use codex_reply(threadId="{info["thread_id"]}", prompt="...") to continue this session.'
            )
        if info.get("worktree_path") or info.get("worktree_branch"):
            wt_lines = []
            if info.get("worktree_path"):
                wt_lines.append(f"Worktree Path: {info['worktree_path']}")
            if info.get("worktree_branch"):
                wt_lines.append(f"Worktree Branch: {info['worktree_branch']}")
            details.append("\n".join(wt_lines))
        if details:
            text += "\n\n---\n" + "\n\n---\n".join(details)
        structured = _clean_none({
            "status": "completed",
            "text": text,
            "thread_id": info.get("thread_id"),
            "model": info.get("model") or meta.get("model"),
            "exit_code": info.get("exit_code"),
            "worktree_path": info.get("worktree_path"),
            "worktree_branch": info.get("worktree_branch"),
        })
        return {
            "resultType": "complete",
            "content": [{"type": "text", "text": text}],
            "structuredContent": structured,
            "isError": False,
        }

    # Tool-level Codex failure (exit != 0 / turn.failed) -> CallToolResult with isError: true
    exit_code = info.get("exit_code")
    exit_display = exit_code if exit_code is not None else "?"
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
    text = f"{label} FAILED (exit {exit_display})\n{detail}"
    failure_obj = _clean_none({
        "model": info.get("model"),
        "cause": info.get("failure_cause") or "Unknown Codex failure",
        "category": info.get("failure_category") or "unknown",
        "retryable": info.get("retryable"),
        "action": info.get("failure_action"),
        "stdout_diagnostics": info.get("stdout_diagnostics"),
        "stderr_diagnostics": info.get("stderr_diagnostics"),
    })
    structured = _clean_none({
        "status": "failed",
        "text": text,
        "thread_id": info.get("thread_id"),
        "model": info.get("model") or meta.get("model"),
        "exit_code": info.get("exit_code"),
        "worktree_path": info.get("worktree_path"),
        "worktree_branch": info.get("worktree_branch"),
        "partial_output": info.get("partial_output"),
        "failure": failure_obj,
    })
    return {
        "resultType": "complete",
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
        "isError": True,
    }


def _send_result(
    rid: Any,
    result: Dict[str, Any],
    *,
    stateless: bool = False,
    force_result_type: Optional[str] = None,
    is_notification: bool = False,
) -> None:
    if is_notification:
        return
    payload = dict(result)
    if force_result_type is not None:
        payload["resultType"] = force_result_type
    elif stateless and "resultType" not in payload:
        payload["resultType"] = "complete"
    _send({"jsonrpc": "2.0", "id": rid, "result": payload})


def _send_error(
    rid: Any,
    code: int,
    message: str,
    *,
    data: Any = None,
    is_notification: bool = False,
) -> None:
    if is_notification:
        return
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    _send({"jsonrpc": "2.0", "id": rid, "error": err})


def _validate_task_ids_param(raw: Any) -> Tuple[Optional[List[str]], Optional[str]]:
    if not isinstance(raw, list) or not raw:
        return None, "task_ids must be a non-empty array of task ID strings"
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            return None, "Every element of task_ids must be a non-empty string"
    return raw, None


def _validate_optional_config(raw: Any) -> Optional[str]:
    if raw is not None and not isinstance(raw, dict):
        return "config must be an object"
    return None


def _validate_optional_thread_id(raw: Any) -> Optional[str]:
    if raw is not None and (not isinstance(raw, str) or not raw.strip()):
        return "threadId must be a non-empty string"
    return None


def _validate_codex_params(args: Dict[str, Any], *, is_reply: bool = False) -> Optional[str]:
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


# ===================================================================
# Request handler
# ===================================================================

def _handle(request: Dict[str, Any]) -> None:
    has_id = "id" in request
    rid = request.get("id")
    if has_id and rid is not None and not _is_valid_request_id(rid):
        _send_error(
            None,
            -32600,
            "Invalid request: id must be a string, number, or null",
        )
        return
    is_notification = (not has_id) or (rid is None)

    method = request.get("method")
    if not isinstance(method, str) or not method:
        _send_error(
            rid,
            -32600,
            "Invalid request: method must be a non-empty string",
            is_notification=is_notification,
        )
        return

    raw_params = request.get("params")
    if raw_params is None:
        params: Dict[str, Any] = {}
    elif isinstance(raw_params, dict):
        params = raw_params
    else:
        _send_error(
            rid,
            -32602,
            "Invalid params: expected object",
            is_notification=is_notification,
        )
        return

    # --- Known notifications (JSON-RPC 2.0: notifications never produce responses) ---
    if method == "notifications/initialized":
        return

    if method == "notifications/cancelled":
        cancelled_id = params.get("requestId")
        if _is_valid_request_id(cancelled_id):
            logging.info("Client cancelled request id=%s", cancelled_id)
            _mark_cancelled(cancelled_id)
            # Kill the child immediately rather than waiting for the next
            # 2s poll -- and, more importantly, so it cannot outlive the
            # request as an untracked orphan.
            _terminate_sync_proc(cancelled_id)
        return

    # JSON-RPC 2.0 §4.1: unknown notifications MUST NOT receive a reply.
    if is_notification:
        logging.debug("Ignoring notification method=%s", method)
        return

    # --- Per-request MCP 2026-07-28 metadata ---
    raw_meta = params.get("_meta")
    req_meta = raw_meta if isinstance(raw_meta, dict) else {}
    proto_ver = req_meta.get("io.modelcontextprotocol/protocolVersion")
    raw_client_caps = req_meta.get("io.modelcontextprotocol/clientCapabilities")
    client_caps = raw_client_caps if isinstance(raw_client_caps, dict) else {}

    if method not in ("server/discover", "initialize") and proto_ver is not None:
        if not isinstance(proto_ver, str) or proto_ver not in _ALL_PROTOCOL_VERSIONS:
            _send_error(
                rid,
                -32022,
                f"Unsupported protocol version: {proto_ver}",
                data={
                    "supportedVersions": list(_ALL_PROTOCOL_VERSIONS),
                    "supported": list(_ALL_PROTOCOL_VERSIONS),
                    "requested": proto_ver,
                },
            )
            return

    is_stateless = proto_ver in _STATELESS_PROTOCOL_VERSIONS

    # --- Discovery & Lifecycle ---
    if method == "server/discover":
        _send_result(
            rid,
            {
                "supportedVersions": list(_ALL_PROTOCOL_VERSIONS),
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "extensions": {
                        _TASKS_EXTENSION_ID: {},
                    },
                },
                "serverInfo": {
                    "name": "codex-mcp-swarm",
                    "version": __version__,
                },
                "instructions": SERVER_INSTRUCTIONS,
                "ttlMs": _DISCOVER_TTL_MS,
                "cacheScope": "public",
                "_meta": {
                    "io.modelcontextprotocol/serverInfo": {
                        "name": "codex-mcp-swarm",
                        "version": __version__,
                    },
                },
            },
            force_result_type="complete",
        )
        return

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
        _send_result(
            rid,
            {
                "protocolVersion": negotiated,
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {
                    "name": "codex-mcp-swarm",
                    "version": __version__,
                },
                "instructions": SERVER_INSTRUCTIONS,
            },
        )
        return

    if method == "ping":
        _send_result(rid, {}, stateless=is_stateless)
        return

    # --- Tools & Resources listing/reading ---
    if method == "tools/list":
        _send_result(rid, {"tools": TOOLS}, stateless=is_stateless)
        return

    if method == "resources/list":
        _send_result(rid, {"resources": RESOURCES}, stateless=is_stateless)
        return

    if method == "resources/read":
        uri = params.get("uri")
        if not isinstance(uri, str) or not uri:
            _send_error(rid, -32602, "uri must be a non-empty string")
            return
        content = _read_resource(uri)
        if content is not None:
            _send_result(
                rid,
                {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": content,
                    }],
                },
                stateless=is_stateless,
            )
        else:
            _send_error(rid, -32602, f"Unknown resource: {uri}")
        return

    # --- Official MCP Tasks extension (io.modelcontextprotocol/tasks) ---
    if method == "tasks/get":
        task_id = params.get("taskId")
        if not isinstance(task_id, str) or not _validate_task_id(task_id):
            _send_error(rid, -32602, f"Invalid taskId: {task_id!r}")
            return
        state = _resolve_task_state(task_id)
        status = state["status"]
        if status == "not_found":
            _send_error(rid, -32602, f"Unknown task: {task_id}")
            return
        if status == "error":
            _send_error(rid, -32603, state.get("error", "Task metadata error"))
            return

        meta = state.get("meta", {})
        started_at = meta.get("started_at", time.time())
        created_at_iso = _iso_utc(started_at)
        ttl_ms = _TASK_MAX_AGE * 1000

        if status == "running":
            stdout_path = TASK_DIR / f"{task_id}.stdout"
            _, jsonl_status = _parse_jsonl_file(stdout_path)
            last_updated = meta.get("last_updated_at", started_at)
            try:
                if stdout_path.exists():
                    last_updated = max(last_updated, stdout_path.stat().st_mtime)
            except OSError:
                pass
            _send_result(
                rid,
                {
                    "taskId": task_id,
                    "status": "working",
                    "statusMessage": _format_mcp_task_status_message(
                        jsonl_status, state["elapsed_seconds"]
                    ),
                    "createdAt": created_at_iso,
                    "lastUpdatedAt": _iso_utc(last_updated),
                    "ttlMs": ttl_ms,
                    "pollIntervalMs": _TASK_POLL_INTERVAL_MS,
                },
                force_result_type="complete",
            )
            return

        info = _check_task(task_id, state=state)
        completed_at = meta.get("completed_at") or meta.get("last_updated_at") or time.time()

        # Per MCP Tasks extension (SEP-2663 / io.modelcontextprotocol/tasks):
        # - Tool execution outcomes -- INCLUDING tool-level errors with
        #   `isError: true` (e.g. Codex exit 1 / turn.failed) -- MUST be
        #   returned under `status: "completed"` with `result: <CallToolResult>`
        #   (`status: "failed"` is strictly reserved for JSON-RPC/protocol errors
        #   carrying an `error` object instead of `result`).
        # - `status: "cancelled"` omits `result` and `error` and conveys
        #   cancellation details via `statusMessage`.
        if status == "cancelled":
            msg_parts = ["Task cancelled"]
            if info.get("worktree_path"):
                msg_parts.append(f"worktree preserved: {info['worktree_path']}")
            if info.get("worktree_branch"):
                msg_parts.append(f"branch: {info['worktree_branch']}")
            _send_result(
                rid,
                {
                    "taskId": task_id,
                    "status": "cancelled",
                    "statusMessage": "; ".join(msg_parts),
                    "createdAt": created_at_iso,
                    "lastUpdatedAt": _iso_utc(completed_at),
                    "ttlMs": ttl_ms,
                },
                force_result_type="complete",
            )
            return

        tool_result = _build_mcp_task_tool_result(info, meta)
        if status == "completed":
            status_message = f"Completed in {info['elapsed_seconds']}s"
        else:
            status_message = info.get("failure_cause") or "Codex task failed"

        _send_result(
            rid,
            {
                "taskId": task_id,
                "status": "completed",
                "statusMessage": status_message,
                "createdAt": created_at_iso,
                "lastUpdatedAt": _iso_utc(completed_at),
                "ttlMs": ttl_ms,
                "result": tool_result,
            },
            force_result_type="complete",
        )
        return

    if method == "tasks/cancel":
        task_id = params.get("taskId")
        if not isinstance(task_id, str) or not _validate_task_id(task_id):
            _send_error(rid, -32602, f"Invalid taskId: {task_id!r}")
            return
        cancel_res = _cancel_task(task_id)
        if cancel_res["status"] == "not_found":
            _send_error(rid, -32602, cancel_res.get("error", f"Unknown task: {task_id}"))
            return
        if cancel_res.get("already_terminal"):
            _send_error(
                rid,
                -32602,
                f"Cannot cancel task {task_id}: task is already in terminal state '{cancel_res['status']}'",
            )
            return
        if cancel_res["status"] == "error":
            _send_error(rid, -32603, cancel_res.get("error", "Failed to cancel task"))
            return
        _send_result(rid, {}, force_result_type="complete")
        return

    # --- Tool calls ---
    if method == "tools/call":
        tool = params.get("name")
        raw_args = params.get("arguments")
        if raw_args is None:
            args: Dict[str, Any] = {}
        elif isinstance(raw_args, dict):
            args = dict(raw_args)
        else:
            _send_error(rid, -32602, "arguments must be an object")
            return

        # Clients that want progress supply a token here; without one the
        # heartbeat is a no-op and long sync runs stay silent as before.
        progress_token = req_meta.get("progressToken")
        use_mcp_tasks = (
            (proto_ver is None or is_stateless)
            and _client_supports_tasks(client_caps, req_meta)
        )

        if tool == "codex":
            prompt = args.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                _send_error(rid, -32602, "prompt must be a non-empty string")
                return
            err_msg = _validate_codex_params(args)
            if err_msg:
                _send_error(rid, -32602, err_msg)
                return

            args.pop("timeout", None)  # ignored: sync tools run to completion

            if use_mcp_tasks:
                try:
                    launch_info = _start_async_internal(
                        args,
                        high_entropy_id=True,
                        mcp_task=True,
                        tool_name="codex",
                    )
                except Exception as exc:
                    err_text = f"Error starting Codex task: {exc}"
                    _send_result(
                        rid,
                        {
                            "content": [{"type": "text", "text": err_text}],
                            "structuredContent": {"status": "failed", "text": err_text},
                            "isError": True,
                        },
                        stateless=True,
                    )
                    return
                now_iso = _iso_utc(launch_info["started_at"])
                _send_result(
                    rid,
                    {
                        "taskId": launch_info["task_id"],
                        "status": "working",
                        "statusMessage": "Starting Codex session...",
                        "createdAt": now_iso,
                        "lastUpdatedAt": now_iso,
                        "ttlMs": _TASK_MAX_AGE * 1000,
                        "pollIntervalMs": _TASK_POLL_INTERVAL_MS,
                    },
                    force_result_type="task",
                )
                return

            outcome = _run_sync_detailed(
                args,
                request_id=rid,
                progress_token=progress_token,
            )
            text = outcome["text"]
            thread_id = outcome["thread_id"]
            worktree_info = outcome.get("worktree_info")
            details = []
            if thread_id:
                details.append(
                    f'Thread ID: {thread_id}\nUse codex_reply(threadId="{thread_id}", prompt="...") to continue this session.'
                )
            if worktree_info:
                details.append(
                    f"Worktree Path: {worktree_info['worktree_path']}\n"
                    f"Worktree Branch: {worktree_info['worktree_branch']}"
                )
            if details:
                text += "\n\n---\n" + "\n\n---\n".join(details)
            structured = _clean_none({
                "status": outcome["status"],
                "text": text,
                "thread_id": thread_id,
                "model": outcome.get("model"),
                "exit_code": outcome.get("exit_code"),
                "worktree_path": worktree_info["worktree_path"] if worktree_info else None,
                "worktree_branch": worktree_info["worktree_branch"] if worktree_info else None,
                "partial_output": outcome.get("partial_output"),
                "failure": _failure_structured(outcome.get("failure")),
            })
            _send_result(
                rid,
                {
                    "content": [{"type": "text", "text": text}],
                    "structuredContent": structured,
                    "isError": bool(outcome["failed"]),
                },
                stateless=is_stateless,
            )

        elif tool == "codex_async":
            prompt = args.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                _send_error(rid, -32602, "prompt must be a non-empty string")
                return
            err_msg = _validate_codex_params(args)
            if err_msg:
                _send_error(rid, -32602, err_msg)
                return

            try:
                launch_info = _start_async(args)
            except Exception as exc:
                err_text = f"Error starting async Codex task: {exc}"
                _send_result(
                    rid,
                    {
                        "content": [{"type": "text", "text": err_text}],
                        "structuredContent": {"status": "error", "error": str(exc)},
                        "isError": True,
                    },
                    stateless=is_stateless,
                )
                return

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
            structured = _clean_none({
                "status": "started",
                "task_id": task_id,
                "model": launch_info.get("model"),
                "worktree_path": launch_info.get("worktree_path"),
                "worktree_branch": launch_info.get("worktree_branch"),
            })
            _send_result(
                rid,
                {
                    "content": [{
                        "type": "text",
                        "text": "\n".join(lines),
                    }],
                    "structuredContent": structured,
                    "isError": False,
                },
                stateless=is_stateless,
            )

        elif tool == "codex_reply":
            thread_id = args.get("threadId")
            prompt = args.get("prompt")

            if (
                not isinstance(thread_id, str)
                or not thread_id.strip()
                or not isinstance(prompt, str)
                or not prompt
            ):
                _send_error(rid, -32602, "threadId and prompt are required strings")
                return
            err_msg = _validate_codex_params(args, is_reply=True)
            if err_msg:
                _send_error(rid, -32602, err_msg)
                return

            extra_params = {
                k: args[k]
                for k in ("cwd", "model", "sandbox", "approval-policy", "config")
                if k in args and args[k] is not None
            }

            if use_mcp_tasks:
                reply_params = dict(extra_params)
                reply_params["threadId"] = thread_id
                reply_params["prompt"] = prompt
                try:
                    launch_info = _start_async_internal(
                        reply_params,
                        high_entropy_id=True,
                        mcp_task=True,
                        tool_name="codex_reply",
                    )
                except Exception as exc:
                    err_text = f"Error starting Codex reply task: {exc}"
                    _send_result(
                        rid,
                        {
                            "content": [{"type": "text", "text": err_text}],
                            "structuredContent": {
                                "status": "failed",
                                "text": err_text,
                                "thread_id": thread_id,
                            },
                            "isError": True,
                        },
                        stateless=True,
                    )
                    return
                now_iso = _iso_utc(launch_info["started_at"])
                _send_result(
                    rid,
                    {
                        "taskId": launch_info["task_id"],
                        "status": "working",
                        "statusMessage": "Starting Codex reply session...",
                        "createdAt": now_iso,
                        "lastUpdatedAt": now_iso,
                        "ttlMs": _TASK_MAX_AGE * 1000,
                        "pollIntervalMs": _TASK_POLL_INTERVAL_MS,
                    },
                    force_result_type="task",
                )
                return

            outcome = _run_reply_sync_detailed(
                thread_id,
                prompt,
                extra_params=extra_params,
                request_id=rid,
                progress_token=progress_token,
            )
            structured = _clean_none({
                "status": outcome["status"],
                "text": outcome["text"],
                "thread_id": outcome.get("thread_id"),
                "model": outcome.get("model"),
                "exit_code": outcome.get("exit_code"),
                "partial_output": outcome.get("partial_output"),
                "failure": _failure_structured(outcome.get("failure")),
            })
            _send_result(
                rid,
                {
                    "content": [{"type": "text", "text": outcome["text"]}],
                    "structuredContent": structured,
                    "isError": bool(outcome["failed"]),
                },
                stateless=is_stateless,
            )

        elif tool == "codex_status":
            task_ids, err_msg = _validate_task_ids_param(args.get("task_ids"))
            if err_msg or task_ids is None:
                _send_error(rid, -32602, err_msg or "task_ids is required")
                return

            parts = []
            structured_tasks = []
            error_count = 0

            for tid in task_ids:
                state = _resolve_task_state(tid)
                status = state["status"]

                if status in ("error", "not_found"):
                    error_count += 1
                    parts.append(f"=== Task {tid} === {state.get('error', status).upper()}")
                    structured_tasks.append(_clean_none({
                        "task_id": tid,
                        "status": status,
                        "error": state.get("error", status),
                    }))
                    continue

                elapsed = state["elapsed_seconds"]
                stdout_path = TASK_DIR / f"{tid}.stdout"
                stderr_path = TASK_DIR / f"{tid}.stderr"
                # Single-pass parse (with incremental cache on large files)
                analysis, jsonl_status = _parse_jsonl_file(stdout_path)

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

                meta = state.get("meta", {})
                tid_thread = meta.get("thread_id") or analysis.get("thread_id")
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

                failure_info: Optional[Dict[str, Any]] = None
                if status == "failed":
                    stdout_text = (
                        ""
                        if (analysis.get("terminal_error") or analysis.get("errors"))
                        else _safe_read(stdout_path)
                    )
                    failure_info = _build_failure_info(
                        stdout_text,
                        _safe_read(stderr_path),
                        command=meta.get("command"),
                        analysis=analysis,
                        model=meta.get("model"),
                    )
                    lines.extend(_format_failure_info(failure_info).splitlines())
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
                structured_tasks.append(_clean_none({
                    "task_id": tid,
                    "status": status,
                    "elapsed_seconds": elapsed,
                    "phase": display_phase,
                    "tools_called": jsonl_status["tools_called"],
                    "thread_id": tid_thread,
                    "model": (failure_info.get("model") if failure_info else None) or meta.get("model"),
                    "worktree_path": meta.get("worktree_path"),
                    "worktree_branch": meta.get("worktree_branch"),
                    "last_tool": jsonl_status.get("last_tool"),
                    "last_tool_args": jsonl_status.get("last_tool_args"),
                    "last_reasoning": jsonl_status.get("last_reasoning"),
                    "last_assistant_text": jsonl_status.get("last_assistant_text"),
                    "last_warning": jsonl_status.get("last_warning"),
                    "last_error": jsonl_status.get("last_error"),
                    "exit_code": state.get("exit_code") if status in ("completed", "failed") else None,
                    "exit_code_lost": bool(state.get("exit_code_lost")) if status in ("completed", "failed") else None,
                    "status_inferred": bool(state.get("status_inferred")) if status in ("completed", "failed") else None,
                    "terminal_conflict": bool(jsonl_status.get("terminal_conflict")) or None,
                    "failure": _failure_structured(failure_info),
                }))

            _send_result(
                rid,
                {
                    "content": [{"type": "text", "text": "\n\n".join(parts)}],
                    "structuredContent": {"tasks": structured_tasks},
                    "isError": error_count == len(task_ids),
                },
                stateless=is_stateless,
            )

        elif tool == "codex_wait":
            task_ids, err_msg = _validate_task_ids_param(args.get("task_ids"))
            if err_msg or task_ids is None:
                _send_error(rid, -32602, err_msg or "task_ids is required")
                return

            timeout = args.get("timeout", 1800) if "timeout" in args else 1800
            if timeout is not None:
                if (
                    isinstance(timeout, bool)
                    or not isinstance(timeout, (int, float))
                    or math.isnan(timeout)
                    or math.isinf(timeout)
                    or timeout <= 0
                ):
                    _send_error(rid, -32602, "timeout must be a positive number or null")
                    return
                # Raise short waits past the client's backgrounding threshold.
                # Finished tasks are unaffected -- they resolve before the deadline.
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
            structured_tasks = []
            any_error = False

            for tid in task_ids:
                info = results.get(tid, {"status": "unknown", "task_id": tid})
                st = info["status"]
                elapsed = info.get("elapsed_seconds", 0)

                if st == "completed":
                    header = f"=== Task {tid} (completed in {elapsed}s) ==="
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
                elif st == "failed":
                    any_error = True
                    exit_code = info.get("exit_code")
                    if exit_code is None:
                        exit_code = "?"
                    header = f"=== Task {tid} (FAILED in {elapsed}s, exit {exit_code}) ==="
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
                elif st == "cancelled":
                    header = f"=== Task {tid} (CANCELLED after {elapsed}s) ==="
                    if info.get("worktree_path"):
                        header += f"\nWorktree: {info['worktree_path']}"
                    if info.get("worktree_branch"):
                        header += f"\nBranch: {info['worktree_branch']}"
                    detail = (
                        info.get("error")
                        if info.get("wait_cancelled")
                        else info.get("result", "Task was cancelled")
                    )
                    parts.append(f"{header}\n{detail}")
                elif st == "timeout":
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
                    any_error = True
                    header = (
                        f"=== Task {tid} === "
                        f"{info.get('error', st)}"
                    )
                    if info.get("worktree_path"):
                        header += f"\nWorktree: {info['worktree_path']}"
                    if info.get("worktree_branch"):
                        header += f"\nBranch: {info['worktree_branch']}"
                    parts.append(header)

                structured_tasks.append(_clean_none({
                    "task_id": tid,
                    "status": st,
                    "elapsed_seconds": info.get("elapsed_seconds"),
                    "result": info.get("result"),
                    "thread_id": info.get("thread_id"),
                    "model": info.get("model"),
                    "worktree_path": info.get("worktree_path"),
                    "worktree_branch": info.get("worktree_branch"),
                    "exit_code": info.get("exit_code"),
                    "exit_code_lost": info.get("exit_code_lost"),
                    "status_inferred": info.get("status_inferred"),
                    "failure_cause": info.get("failure_cause"),
                    "failure_category": info.get("failure_category"),
                    "retryable": info.get("retryable"),
                    "failure_action": info.get("failure_action"),
                    "partial_output": info.get("partial_output"),
                    "stdout_diagnostics": info.get("stdout_diagnostics"),
                    "stderr_diagnostics": info.get("stderr_diagnostics"),
                    "error": info.get("error"),
                }))

            _send_result(
                rid,
                {
                    "content": [{"type": "text", "text": "\n\n".join(parts)}],
                    "structuredContent": {"tasks": structured_tasks},
                    "isError": any_error,
                },
                stateless=is_stateless,
            )

        elif tool == "codex_cancel":
            task_id = args.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                _send_error(rid, -32602, "task_id must be a non-empty string")
                return

            result = _cancel_task(task_id)
            lines = [f"Task {task_id}: {result.get('message', result.get('error', result['status']))}"]
            if result.get("worktree_path"):
                lines.append(f"Worktree preserved: {result['worktree_path']}")
            if result.get("worktree_branch"):
                lines.append(f"Branch preserved: {result['worktree_branch']}")
            if result.get("failure_detail"):
                lines.extend(["", result["failure_detail"]])
            structured = _clean_none({
                "task_id": task_id,
                "status": result["status"],
                "message": result.get("message"),
                "error": result.get("error"),
                "worktree_path": result.get("worktree_path"),
                "worktree_branch": result.get("worktree_branch"),
                "exit_code": result.get("exit_code"),
                "exit_code_lost": result.get("exit_code_lost"),
                "status_inferred": result.get("status_inferred"),
                "model": result.get("model"),
                "failure_cause": result.get("failure_cause"),
                "failure_category": result.get("failure_category"),
                "retryable": result.get("retryable"),
                "failure_action": result.get("failure_action"),
            })
            _send_result(
                rid,
                {
                    "content": [{"type": "text", "text": "\n".join(lines)}],
                    "structuredContent": structured,
                    "isError": result["status"] in ("error", "not_found"),
                },
                stateless=is_stateless,
            )

        else:
            _send_error(rid, -32601, f"Unknown tool: {tool}")
        return

    _send_error(rid, -32601, f"Method not found: {method}")

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
_BLOCKING_METHODS = {"tools/call", "tasks/cancel"}


def _handle_threaded(request: Dict[str, Any]) -> None:
    """Wrapper for _handle that catches exceptions in worker threads."""
    rid = request.get("id")
    is_notification = ("id" not in request) or (rid is None)
    safe_rid = rid if _is_valid_request_id(rid) else None
    try:
        _handle(request)
    except Exception as exc:
        logging.error(
            "Handler error (thread): %s\n%s", exc, traceback.format_exc()
        )
        _send_error(
            safe_rid,
            -32603,
            f"Internal error: {exc}",
            is_notification=is_notification,
        )


def main() -> None:
    _parse_args()
    if _IS_WINDOWS:
        # MCP stdio is UTF-8 with \n framing; Windows pipes default to the ANSI
        # code page and print() would emit \r\n.
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    logging.info(
        "Server starting -- defaults=%s flags=%s", SERVER_CONFIG, SERVER_FLAGS
    )

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request: Any = None
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
                method = request.get("method")
                logging.debug("Request: method=%s id=%s", method, request.get("id"))

                if isinstance(method, str) and method in _BLOCKING_METHODS:
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
                is_notification = (
                    isinstance(request, dict)
                    and (("id" not in request) or (request.get("id") is None))
                )
                raw_rid = request.get("id") if isinstance(request, dict) else None
                safe_rid = raw_rid if _is_valid_request_id(raw_rid) else None
                logging.error(
                    "Handler error: %s\n%s", exc, traceback.format_exc()
                )
                _send_error(
                    safe_rid,
                    -32603,
                    f"Internal error: {exc}",
                    is_notification=is_notification,
                )
    except KeyboardInterrupt:
        pass
    finally:
        # Async tasks are detached on purpose (start_new_session) and survive;
        # sync children belong to a request that no longer exists.
        _terminate_all_sync_procs()

    logging.info("Server stopped")


if __name__ == "__main__":
    main()

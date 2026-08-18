# codex-mcp-swarm

An MCP server that wraps OpenAI's Codex CLI with **true parallel execution** and **live task monitoring**. Zero dependencies -- single Python file, stdlib only.

## Why?

The official `codex mcp-server` processes requests sequentially. If your MCP client (Claude Code, etc.) needs to run 5 Codex tasks, they queue up one after another. This server spawns each task as an independent subprocess, so they run in parallel.

**Unique features no other Codex MCP wrapper has:**

- **Worktree isolation** -- `worktree: true` creates an isolated git worktree per task so parallel Codex instances never edit past each other
- **Batch wait** -- launch N tasks, call `codex_wait` once, get all results when they finish
- **Live status** -- see what each Codex task is doing right now (last tool call, current reasoning, progress)
- **Full flag parity** -- same parameters as the official Codex MCP tool (`sandbox`, `approval-policy`, `cwd`, `model`, `config`, etc.)
- **Drop-in config** -- accepts the same `-c key=value` server args as `codex mcp-server`

## Tools

| Tool | Description |
|------|-------------|
| `codex` | Synchronous execution (drop-in replacement for official) |
| `codex_async` | Launch a task, get a `task_id` immediately (fan-out). **Not** fire-and-forget: results come back only via `codex_wait`. Pass `threadId` to resume a session |
| `codex_reply` | Continue a previous session via `codex exec resume` |
| `codex_status` | Live view: tools called, last command, current thinking |
| `codex_wait` | Block until multiple tasks complete, return all results |
| `codex_cancel` | Kill a running async task (preserves worktree for inspection) |

## Installation

### Claude Code

```bash
claude mcp add codex-swarm -- uvx --upgrade codex-mcp-swarm \
  -c model=gpt-5.4 \
  -c approval_policy=never \
  -c sandbox_mode=danger-full-access \
  --skip-git-repo-check
```

That's it. No clone, no setup. `uvx` downloads and runs it directly from PyPI. The `--upgrade` flag ensures you always get the latest version on restart.

> **Note:** Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). Alternatively, use `pipx run codex-mcp-swarm` instead of `uvx codex-mcp-swarm`.

### Manual (`~/.claude.json`)

```json
{
  "mcpServers": {
    "codex-swarm": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--upgrade",
        "codex-mcp-swarm",
        "-c", "model=gpt-5.4",
        "-c", "approval_policy=never",
        "-c", "sandbox_mode=danger-full-access",
        "--skip-git-repo-check"
      ]
    }
  }
}
```

The `-c` flags are identical to `codex mcp-server` -- copy-paste your existing config.

## Usage

### Parallel execution

```
1. Call codex_async with prompt A  -->  task_id: "abc123"
2. Call codex_async with prompt B  -->  task_id: "def456"
3. Call codex_async with prompt C  -->  task_id: "ghi789"
4. Call codex_wait(task_ids=["abc123", "def456", "ghi789"])
   --> blocks until all finish, returns all results
```

### Collecting results: read this before you fan out

A `task_id` is internal to this server. **Your MCP client does not track it.**
It will not appear in a task list, and no notification fires when the task
finishes. If the agent's turn ends before `codex_wait` is called, the Codex run
still completes and writes its result to disk, but nothing is left to deliver
it and nothing wakes the session. `codex_status` is a progress peek only; it
collects nothing.

So: **`codex_async` and `codex_wait` belong in the same turn.**

Let the wait run long. Claude Code moves any main-conversation tool call still
running after two minutes into a tracked background task and re-invokes the
session with the result when it settles -- that is the wake-up mechanism, so
crossing the two-minute line is the goal rather than something to dodge. Short
`timeout` values are raised to a floor (default 150s, override with
`CODEX_SWARM_MIN_WAIT`) for exactly this reason. Tasks that have already
finished return instantly regardless of the floor.

### A reported failure is not proof of failure

`codex exec` writes progress, warnings and sandbox notices to stderr on
perfectly healthy runs. Until 1.10.0, a task whose exit code was lost to a
reaping race was marked **failed** on the strength of non-empty stderr alone,
so finished work got reported as a failure and callers acted on it.

The verdict now comes from the run's own output: a completed `agent_message`
in the JSONL means the run reached the end, whatever stderr says. When the
exit code really was lost, the result carries `exit_code_lost` and the output
says the verdict is inferred rather than observed, along with what to check.

Two things follow for anyone consuming this server:

- A `FAILED` from `codex_status` or `codex_wait` that is flagged as inferred is
  a prompt to verify, not a conclusion. Check `pgrep -af 'codex exec'`, file
  mtimes, the worktree branch, and your own build or tests.
- A `codex_wait` that times out is **not** a failure at all. The task is not
  killed. Call `codex_wait` again with the same `task_id`.

### Worktree isolation

Prevent parallel tasks from editing the same files:

```
1. Call codex_async(prompt="Refactor auth", worktree=true)
   --> task_id: "abc123"
   --> Worktree Branch: codex-swarm/abc123

2. Call codex_async(prompt="Add logging", worktree=true)
   --> task_id: "def456"
   --> Worktree Branch: codex-swarm/def456

3. codex_wait(task_ids=["abc123", "def456"])
4. git merge codex-swarm/abc123
5. git merge codex-swarm/def456
```

Each task gets its own git worktree and branch based on HEAD. After completion, merge the branches back. Worktrees are automatically cleaned up after 24 hours (configurable via `CODEX_SWARM_TASK_MAX_AGE`).

### Live monitoring

```
Call codex_status(task_ids=["abc123"])
-->
=== Task abc123 (45s elapsed) ===
Phase: running
Tools called: 23
Last tool: exec_command(grep -rn "handleError" src/)
Output: Analyzing error handling patterns across the codebase...
```

### Session continuity

```
1. Call codex(prompt="Review this file")  -->  result + session persisted
2. Call codex_reply(threadId="<session-uuid>", prompt="Now fix the bug you found")
```

For a follow-up that will run more than a few minutes, resume in the
background instead -- `codex_async` accepts `threadId` and gives you a
`task_id` that survives a client idle timeout:

```
1. Call codex_async(threadId="<session-uuid>", prompt="Now implement it")
                                          -->  task_id: "abc123"
2. Call codex_wait(task_ids=["abc123"])   -->  result, with full prior context
```

`codex_reply` is synchronous and emits progress notifications while it waits,
but a client that gives up on silence will still abandon the request. Only the
async path leaves you a handle to recover with.

## Server flags

| Flag | Description |
|------|-------------|
| `-c key=value` | Config default (repeatable). Same format as `codex mcp-server`. |
| `--skip-git-repo-check` | Allow running outside git repos. |
| `--ephemeral` | Don't persist session files. Disables `codex_reply`. |

## Per-call parameters

All parameters from the official Codex MCP tool are supported:

- `prompt` (required)
- `model` -- override server default
- `sandbox` -- `read-only`, `workspace-write`, `danger-full-access`
- `approval-policy` -- `untrusted`, `on-failure`, `on-request`, `never`
- `cwd` -- working directory
- `profile` -- config profile from `config.toml` (ignored on `threadId` resumes -- `codex exec resume` has no `--profile` flag)
- `config` -- object of key=value overrides
- `worktree` -- run in an isolated git worktree (prevents parallel tasks from conflicting)
- `base-instructions`, `developer-instructions`, `compact-prompt`

## MCP Resources

The server exposes read-only resources for discoverability:

| URI | Description |
|-----|-------------|
| `codex-swarm:///server-info` | Version, capabilities, directories, config |
| `codex-swarm:///config` | Current server-level defaults and flags |
| `codex-swarm:///tasks` | All known tasks and their current state |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEX_SWARM_LOG` | `/tmp/codex_mcp_swarm.log` | Log file path |
| `CODEX_SWARM_LOG_LEVEL` | `WARNING` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CODEX_SWARM_TASK_DIR` | `/tmp/codex_swarm_tasks` | Task output storage directory |
| `CODEX_SWARM_WORKTREE_DIR` | `/tmp/codex-swarm-worktrees` | Worktree storage directory |
| `CODEX_SWARM_TASK_MAX_AGE` | `86400` (24h) | Seconds before completed task artifacts (and worktrees) are cleaned up |
| `CODEX_SWARM_MIN_WAIT` | `150` | Floor in seconds for `codex_wait`. Keeps the call above the client's 2-minute auto-backgrounding threshold so completion re-invokes the session. Already-finished tasks ignore it |

## Requirements

- Python 3.8+
- [Codex CLI](https://github.com/openai/codex) installed and authenticated
- No pip dependencies (stdlib only)
- Works on Linux and macOS (Linux gets extra PID reuse protection and zombie detection via `/proc`)

## Credits

Originally inspired by [jeanchristophe13v/codex-mcp-async](https://github.com/jeanchristophe13v/codex-mcp-async). Rewritten with full flag parity, JSONL status parsing, batch wait, and session reply support.

## License

MIT

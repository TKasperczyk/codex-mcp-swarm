# codex-mcp-swarm

An MCP server that wraps OpenAI's Codex CLI with **true parallel execution** and **live task monitoring**. Zero dependencies -- single Python file, stdlib only.

## Why?

The official `codex mcp-server` processes requests sequentially. If your MCP client (Claude Code, etc.) needs to run 5 Codex tasks, they queue up one after another. This server spawns each task as an independent subprocess, so they run in parallel.

**Unique features no other Codex MCP wrapper has:**

- **Batch wait** -- launch N tasks, call `codex_wait` once, get all results when they finish
- **Live status** -- see what each Codex task is doing right now (last tool call, current reasoning, progress)
- **Full flag parity** -- same parameters as the official Codex MCP tool (`sandbox`, `approval-policy`, `cwd`, `model`, `config`, etc.)
- **Drop-in config** -- accepts the same `-c key=value` server args as `codex mcp-server`

## Tools

| Tool | Description |
|------|-------------|
| `codex` | Synchronous execution (drop-in replacement for official) |
| `codex_async` | Fire-and-forget -- returns a `task_id` immediately |
| `codex_reply` | Continue a previous session via `codex exec resume` |
| `codex_status` | Live view: tools called, last command, current thinking |
| `codex_wait` | Block until multiple tasks complete, return all results |

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
- `profile` -- config profile from `config.toml`
- `config` -- object of key=value overrides
- `base-instructions`, `developer-instructions`, `compact-prompt`

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEX_SWARM_LOG` | `/tmp/codex_mcp_swarm.log` | Log file path |
| `CODEX_SWARM_LOG_LEVEL` | `WARNING` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CODEX_SWARM_TASK_DIR` | `/tmp/codex_swarm_tasks` | Task output storage directory |

## Requirements

- Python 3.8+
- [Codex CLI](https://github.com/openai/codex) installed and authenticated
- No pip dependencies (stdlib only)
- Works on Linux and macOS (Linux gets extra PID reuse protection and zombie detection via `/proc`)

## Credits

Originally inspired by [jeanchristophe13v/codex-mcp-async](https://github.com/jeanchristophe13v/codex-mcp-async). Rewritten with full flag parity, JSONL status parsing, batch wait, and session reply support.

## License

MIT

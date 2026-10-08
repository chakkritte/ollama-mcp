# Ollama MCP Server (with sub-agent support)

A lightweight MCP server that exposes Ollama Cloud models to MCP clients such as
Claude Code. Besides single-shot chat and web tools, it can run an Ollama model
as an **autonomous sub-agent** (tool-calling loop) and return a final report.

## Requirements

- Python 3.10+
- An Ollama Cloud account and API key

## Install

```bash
git clone https://github.com/chakkritte/ollama-mcp.git
cd ollama-mcp
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then edit OLLAMA_API_KEY
```

## Claude Code setup

Copy `.mcp.json.example` to your project as `.mcp.json` (or add the entry to
your MCP config) and fix the absolute paths. Copy
`.claude/agents/ollama-worker.md` into your project's `.claude/agents/` to get a
ready-made dispatcher sub-agent. Claude Code can then run `ollama-worker`
in parallel with its own work.

## Tools

| Tool | Purpose |
|------|---------|
| `list_models` | List models on your Ollama Cloud account |
| `get_current_model` / `set_current_model` | Read / change the process-wide default model (prefer passing `model` per call) |
| `chat` | One prompt to a model. Options: `system`, `model`, `think` (`true`/`low`/`medium`/`high`), `history` (prior messages, client-managed) |
| `web_search` / `web_fetch` | Ollama hosted web APIs |
| `agent_run` | **Run a model as a sub-agent** with tools, return its final report |
| `agent_tools` | Show agent tool groups and which are enabled |

### `agent_run`

| Argument | Meaning |
|----------|---------|
| `task` | Self-contained brief; the agent sees nothing else |
| `model` | Must support tool calling (e.g. `gpt-oss:120b`, `kimi-k3`) |
| `tools` | Groups: `read`, `web`, `write`, `shell` (default `["read","web"]`) |
| `system` | Extra role instructions |
| `max_steps` | Model turns, capped by `OLLAMA_AGENT_MAX_STEPS` (default 20) |
| `timeout_seconds` | Wall-clock limit for the run |
| `think` | Optional thinking mode |
| `workspace` | Sub-directory of the workspace root to confine the agent to |

Returns `status` (`completed` / `max_steps` / `timeout` / `error`), `content`
(final report), `steps`, and `tool_calls` (an audit trail of what the agent did).

| Group | Tools |
|-------|-------|
| `read` | `read_file`, `list_dir`, `grep` |
| `web` | `web_search`, `web_fetch` |
| `write` | `write_file`, `edit_file` |
| `shell` | `run_command` |

## Security model

The remote model is treated as untrusted.

- All paths are confined to `OLLAMA_AGENT_WORKSPACE` (default: the server's
  working directory). `..` escapes, symlink escapes, `.env*` files and `.git`
  internals are blocked, so the API key cannot be read through file tools.
- `write` and `shell` are **off by default**. They need
  `OLLAMA_AGENT_ALLOW_WRITE=1` / `OLLAMA_AGENT_ALLOW_SHELL=1` on the server
  **and** an explicit `tools` entry in the call.
- `run_command` does not use a shell (pipes, `;`, redirects are inert), must
  match `OLLAMA_AGENT_SHELL_ALLOWLIST` token-by-token, has a timeout, and runs
  without `OLLAMA_API_KEY` in its environment. Note that allow-listing `pytest`
  executes project code; trim the list if that is not acceptable.
- Everything you send in `task` and every file the agent reads is sent to
  Ollama Cloud. Do not point it at private data you would not send there.
- Agent output is unverified model output; review changes (`git diff`) before
  keeping them.

## Concurrency

All tools are async. Up to `OLLAMA_AGENT_MAX_CONCURRENT` (default 4) agents run
at once; extra calls wait for a slot. Always pass `model` explicitly when running
agents in parallel, so `set_current_model` cannot affect them.

## Configuration

See `.env.example`. Main variables: `OLLAMA_HOST`, `OLLAMA_API_KEY`,
`DEFAULT_MODEL`, `OLLAMA_AGENT_WORKSPACE`, `OLLAMA_AGENT_ALLOW_WRITE`,
`OLLAMA_AGENT_ALLOW_SHELL`, `OLLAMA_AGENT_SHELL_ALLOWLIST`,
`OLLAMA_AGENT_MAX_CONCURRENT`, `OLLAMA_AGENT_MAX_STEPS`, `OLLAMA_AGENT_TIMEOUT`.

## Model guide

Based on the Ollama cloud catalog (https://ollama.com/search?c=cloud, checked
2026-10-08). The catalog changes often; re-check before relying on it. Picks
are my reading of each model's description and tags on that page, **not
benchmark results**, and the page does not list parameter counts for most
cloud models.

### Best pick by task

| Task | Pick | Why (from the catalog) | Alternative |
|------|------|------------------------|-------------|
| Coding / long agentic coding | `glm-5.3` | "most capable open-weights coding model", 1M context | `kimi-k2.7-code` (lower thinking-token use), `minimax-m3` |
| Reasoning / math | `deepseek-v4-pro` | Frontier MoE with three reasoning modes | `kimi-k3` |
| General agentic / orchestration | `kimi-k3` | Moonshot's most capable, multimodal, 1M context | `mistral-large-4`, `nemotron-3-ultra` |
| Multimodal (vision) | `glm-5.3-flash` | Natively multimodal, near top-tier coding | `kimi-k3`, `mistral-large-4` |
| Audio input | `gemma4` | Only cloud model tagged with audio | none |
| Search / fast and cheap | `deepseek-v4.1-flash` | Search-focused, fast, vision + tools | `glm-5.3-flash` |
| Multi-agent / high throughput | `nemotron-3-ultra` | Built for high-throughput reasoning and long-running agents | `nemotron-3-super` |

### Best pick by size

Only some models list sizes, so this table covers those. Which sizes your
account can actually pull may differ from the catalog page: on 2026-10-08 one
account's `list_models` showed only `gemma4:31b`, `gpt-oss:20b`/`120b`,
`nemotron-3-nano:30b` and `nemotron-3-super` (no `:4b`, no smaller `gemma4`).
Run `list_models` before relying on a tiny-size pick.

| Size class | Pick | Notes |
|------------|------|-------|
| Tiny (about 2-4B) | `nemotron-3-nano:4b` | Agentic, tools + thinking. `gemma4:e2b` / `e4b` if you need vision or audio |
| Small (about 12-31B) | `gemma4:31b` (`12b`, `26b` also available) | Vision, tools, thinking, audio, 256K context |
| Mid (20-30B) | `gpt-oss:20b` or `nemotron-3-nano:30b` | `gpt-oss` has the longest track record here; the Nemotron has 256K context |
| Large (about 120B) | `gpt-oss:120b` or `nemotron-3-super:120b` | `nemotron-3-super` is 120B MoE with 12B active, aimed at multi-agent use. `gpt-oss:120b` has 128K context |
| Frontier (size not listed) | `glm-5.3`, `kimi-k3`, `deepseek-v4-pro` | Pick by task above |

### For `agent_run`

Use a model tagged **tools** on the catalog page. All models in the tables
above have it. Good defaults: `glm-5.3` (coding), `kimi-k3` (general),
`gpt-oss:120b` (reliable and cheap), `deepseek-v4.1-flash` (fast research).
`DEFAULT_MODEL` in `.env.example` is still `gpt-oss:120b`; change it if you
prefer another default. If a model rejects tool calling, the run returns
`status: "error"` with the provider's message.

## Tests

```bash
pip install pytest pytest-asyncio
pytest
```

The tests use a scripted fake client, so they need no network or API key.
They do not cover real Ollama Cloud behaviour (see below).

## License

Apache-2.0 (see `LICENSE`).

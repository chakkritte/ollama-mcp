# Ollama MCP Server

A lightweight [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that exposes [Ollama Cloud](https://ollama.com/) models as tools to MCP clients such as Claude Code.

This server not only lets you chat with Ollama Cloud models — it also makes it easy to switch between the best model for each task: coding, math, reasoning, agentic workflows, and more.

## Features

- **List models** – Discover all models available on your Ollama Cloud instance.
- **Get / set default model** – Switch the active model at runtime to match the task.
- **Chat** – Send prompts to any Ollama Cloud model, with optional system messages and per-request model overrides.
- **Task-aware model guide** – Built-in recommendations for choosing the right model for coding, math, reasoning, and other workloads.

## Requirements

- Python 3.10+
- An [Ollama Cloud](https://ollama.com/) account with an API key.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/chakkritt/ollama-mcp.git
   cd ollama-mcp
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
```

Then edit `.env` and replace the placeholder values with your own:

```bash
OLLAMA_HOST=https://ollama.com
OLLAMA_API_KEY=YOUR_OLLAMA_API_KEY
DEFAULT_MODEL=gpt-oss:120b
```

- `OLLAMA_HOST` – Base URL of your Ollama Cloud instance.
- `OLLAMA_API_KEY` – Your Ollama Cloud API key.
- `DEFAULT_MODEL` – Fallback model used when a chat request does not specify one.

You can override the model on every chat request, so the `DEFAULT_MODEL` is just a sensible starting point.

## Model selection guide

Use this guide to pick the best Ollama Cloud model for your task. Switch models anytime with the `set_current_model` tool.

### Coding & software engineering

These models excel at writing, debugging, and reasoning about code.

| Model | Why use it |
|-------|------------|
| `glm-5.2` | Flagship long-horizon model; great for complex coding and agentic engineering tasks. |
| `kimi-k2.7-code` | Coding-specialized variant of Kimi K2.6; efficient long-horizon coding with lower thinking-token usage. |
| `kimi-k2.6` | Native multimodal agentic model with strong long-horizon coding and autonomous execution. |
| `minimax-m3` | Coding and agentic frontier model with a 1M context window and native multimodality. |

**Recommended default for coding:** `glm-5.2` or `kimi-k2.7-code`.

### Math, reasoning & problem solving

| Model | Why use it |
|-------|------------|
| `deepseek-v4-pro` | Frontier MoE with multiple reasoning modes and a large context window; best for deep math and logic. |
| `deepseek-v4-flash` | Fast, efficient 284B MoE (13B activated) with 1M-token context and strong reasoning. |
| `glm-5.2` | Handles long-horizon reasoning tasks well. |
| `nemotron-3-super` | NVIDIA's efficient open MoE for complex multi-agent and reasoning applications. |
| `nemotron-3-ultra` | High-throughput reasoning for extended agent workflows. |

**Recommended default for math:** `deepseek-v4-pro`.

### General-purpose & agentic tasks

| Model | Why use it |
|-------|------------|
| `gpt-oss:120b` | OpenAI open-weight model for reasoning, agentic tasks, and developer use cases. |
| `mistral-large-3` | Production-grade multimodal MoE for enterprise workloads. |
| `qwen3.5` | Large open-source multimodal family with sizes from 0.8B to 122B. |
| `kimi-k2.5` | Multimodal agentic model with vision/language understanding and instant/thinking modes. |

### Multimodal & vision

| Model | Why use it |
|-------|------------|
| `gemma4` | Strong performance across scales; supports reasoning, coding, multimodal understanding, and audio. |
| `gemini-3-flash-preview` | Fast, cost-efficient frontier intelligence. |
| `kimi-k2.6` / `kimi-k2.5` | Native multimodal agentic capabilities. |
| `minimax-m3` | Native multimodality with a 1M context window. |

### Fast / cost-efficient tasks

| Model | Why use it |
|-------|------------|
| `deepseek-v4-flash` | Large MoE with only 13B activated parameters; fast and efficient. |
| `gemini-3-flash-preview` | Optimized for speed and cost. |
| `nemotron-3-nano` | Small, efficient agentic models at 4B and 30B. |
| `qwen3.5:0.8b` / `qwen3.5:2b` | Tiny, fast variants for simple tasks. |

## Running the server

```bash
python server.py
```

By default, the server runs as an MCP server over stdio, which is the standard transport for most MCP clients.

## Available tools

| Tool | Description |
|------|-------------|
| `list_models` | List all available Ollama Cloud models. |
| `get_current_model` | Return the current default model. |
| `set_current_model` | Change the default model used for chat. |
| `chat` | Send a prompt to the current (or specified) model. Supports an optional `system` message and `model` override. |

## Example usage with Claude Code

Add the server to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "ollama": {
      "command": "python",
      "args": ["/absolute/path/to/server.py"],
      "env": {
        "OLLAMA_HOST": "https://ollama.com",
        "OLLAMA_API_KEY": "YOUR_OLLAMA_API_KEY",
        "DEFAULT_MODEL": "gpt-oss:120b"
      }
    }
  }
}
```

Then you can ask Claude to:

- **"List available Ollama models"**
- **"Switch to the coding model"** — the guide recommends `glm-5.2` or `kimi-k2.7-code`.
- **"Use the math model to solve this problem"** — the guide recommends `deepseek-v4-pro`.
- **"Chat with `glm-5.2` and refactor this function"**
- **"Set the default model to `deepseek-v4-pro`"**

## Quick task-to-model cheatsheet

| Task | Suggested model |
|------|-----------------|
| General coding | `glm-5.2`, `kimi-k2.7-code` |
| Complex / long-horizon coding | `glm-5.2`, `kimi-k2.6`, `minimax-m3` |
| Math & deep reasoning | `deepseek-v4-pro`, `deepseek-v4-flash` |
| Fast reasoning on a budget | `deepseek-v4-flash`, `gemini-3-flash-preview` |
| Agentic workflows | `glm-5.2`, `minimax-m3`, `nemotron-3-super` |
| Multimodal tasks | `gemma4`, `kimi-k2.6`, `minimax-m3`, `qwen3.5` |
| General-purpose chat | `gpt-oss:120b`, `mistral-large-3`, `qwen3.5`, `gemma4` |

## Project structure

```
ollama-mcp/
├── .env.example      # Environment variable template
├── .gitignore        # Files ignored by Git
├── LICENSE           # MIT license
├── pyproject.toml    # Project metadata
├── README.md         # This file
├── requirements.txt  # Python dependencies
└── server.py         # MCP server implementation
```

## Dependencies

- [`fastmcp`](https://github.com/jlowin/fastmcp) – Framework for building MCP servers in Python.
- [`ollama`](https://github.com/ollama/ollama-python) – Official Ollama Python client.
- [`python-dotenv`](https://github.com/theskumar/python-dotenv) – Load environment variables from `.env`.

## Keeping model recommendations up to date

Ollama Cloud's model catalog changes frequently. To refresh this guide:

1. Visit https://ollama.com/search?c=cloud.
2. Update the **Model selection guide** and **Quick task-to-model cheatsheet** sections with new releases or benchmarks.
3. Adjust your `DEFAULT_MODEL` and `set_current_model` calls to match your current workload.

## License

Apache License 2.0

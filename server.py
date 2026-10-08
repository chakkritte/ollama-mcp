"""Ollama Cloud MCP server with sub-agent support for Claude Code.

Tools
-----
Model management : list_models, get_current_model, set_current_model
Single-shot      : chat (optional history, timeout, error handling)
Web              : web_search, web_fetch
Sub-agent        : agent_run   <- runs a tool-calling loop on an Ollama model
                   agent_tools <- describes the tool groups agent_run can use

agent_run lets an Ollama model act as a sub-agent: it can read/search files,
search/fetch the web and (opt-in) write files and run allow-listed commands,
then returns a final report to the calling client (e.g. Claude Code).

Safety model (all enforced server-side, the remote model is untrusted):
  * Every file path is confined to the workspace root; .env* files are blocked.
  * The "write" and "shell" groups must be enabled by BOTH the server
    environment (OLLAMA_AGENT_ALLOW_WRITE / OLLAMA_AGENT_ALLOW_SHELL) and the
    individual agent_run call.
  * Shell commands run without a shell (no pipes/redirects), only if they match
    an allow-list, with a timeout, and without the API key in their environment.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Literal

import httpx
import ollama
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com").rstrip("/")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-oss:120b")

WORKSPACE_ROOT = Path(os.getenv("OLLAMA_AGENT_WORKSPACE", os.getcwd())).resolve()
REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120"))  # per model call
AGENT_TIMEOUT = float(os.getenv("OLLAMA_AGENT_TIMEOUT", "600"))  # whole agent run
MAX_CONCURRENT_AGENTS = int(os.getenv("OLLAMA_AGENT_MAX_CONCURRENT", "4"))
MAX_STEPS_CAP = int(os.getenv("OLLAMA_AGENT_MAX_STEPS", "20"))
MAX_TOOL_OUTPUT = int(os.getenv("OLLAMA_AGENT_MAX_TOOL_OUTPUT", "20000"))
MAX_FILE_BYTES = 1_000_000
SHELL_TIMEOUT = float(os.getenv("OLLAMA_AGENT_SHELL_TIMEOUT", "60"))

ALLOW_WRITE = os.getenv("OLLAMA_AGENT_ALLOW_WRITE", "0") == "1"
ALLOW_SHELL = os.getenv("OLLAMA_AGENT_ALLOW_SHELL", "0") == "1"
DEFAULT_SHELL_ALLOWLIST = (
    "ls,cat,head,tail,wc,grep,rg,git status,git diff,git log,git show,"
    "pytest,python -m pytest"
)
SHELL_ALLOWLIST = [
    shlex.split(item)
    for item in os.getenv("OLLAMA_AGENT_SHELL_ALLOWLIST", DEFAULT_SHELL_ALLOWLIST).split(",")
    if item.strip()
]

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache"}

current_model: str = DEFAULT_MODEL
_agent_slots = asyncio.Semaphore(MAX_CONCURRENT_AGENTS)

mcp = FastMCP("Ollama Cloud")


class ToolError(Exception):
    """Raised by agent tools; the message is returned to the model."""


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}


def _client() -> ollama.AsyncClient:
    return ollama.AsyncClient(host=OLLAMA_HOST, headers=_headers(), timeout=REQUEST_TIMEOUT)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or an attribute-style response object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# --------------------------------------------------------------------------- #
# Model management
# --------------------------------------------------------------------------- #


async def list_models() -> dict[str, Any]:
    """List models available on Ollama Cloud."""
    try:
        resp = await _client().list()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    names = [_get(m, "model") or _get(m, "name") for m in _get(resp, "models", [])]
    return {"models": sorted(n for n in names if n), "current_model": current_model}


def get_current_model() -> dict[str, str]:
    """Return the default model used when a call omits `model`."""
    return {"current_model": current_model}


def set_current_model(model: str) -> dict[str, str]:
    """Change the default model (process-wide). Prefer passing `model` per call."""
    global current_model
    current_model = model
    return {"current_model": current_model}


# --------------------------------------------------------------------------- #
# Single-shot chat
# --------------------------------------------------------------------------- #


async def chat(
    prompt: str,
    model: str | None = None,
    system: str = "",
    think: bool | Literal["low", "medium", "high"] | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Send one prompt to an Ollama Cloud model.

    `history` is an optional list of prior {"role", "content"} messages so the
    caller can continue a conversation (the server itself keeps no state).
    """
    used = model or current_model
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(history or [])
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {"model": used, "messages": messages}
    if think is not None:
        kwargs["think"] = think
    try:
        resp = await _client().chat(**kwargs)
    except Exception as exc:  # noqa: BLE001
        return {"model": used, "error": f"{type(exc).__name__}: {exc}"}

    msg = _get(resp, "message", {})
    content = _get(msg, "content", "") or ""
    header = f"[model: {used} | think: {think}]"
    return {
        "model": used,
        "think": think,
        "thinking_enabled": think is not None and think is not False,
        "content": f"{header}\n{content}" if content else header,
        "thinking": _get(msg, "thinking") or None,
    }


# --------------------------------------------------------------------------- #
# Web tools (shared by chat users and by agents)
# --------------------------------------------------------------------------- #


async def _post_ollama(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=30.0, headers=_headers()) as http:
            r = await http.post(f"{OLLAMA_HOST}{path}", json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as exc:
        return {"error": f"HTTP {exc.response.status_code}", "detail": exc.response.text[:500]}
    except httpx.RequestError as exc:
        return {"error": "request failed", "detail": str(exc)}


async def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web via Ollama's hosted API (max 10 results)."""
    data = await _post_ollama(
        "/api/web_search", {"query": query, "max_results": max(1, min(max_results, 10))}
    )
    if "error" in data:
        return data
    return {
        "results": [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": (r.get("content") or r.get("snippet") or "")[:500],
            }
            for r in data.get("results", [])
        ]
    }


async def web_fetch(url: str) -> dict[str, Any]:
    """Fetch a page via Ollama's hosted API (title, main text, up to 10 links)."""
    data = await _post_ollama("/api/web_fetch", {"url": url})
    if "error" in data:
        return data
    return {
        "title": data.get("title"),
        "content": data.get("content"),
        "links": (data.get("links") or [])[:10],
    }


# --------------------------------------------------------------------------- #
# Agent tool implementations (sandboxed to the workspace)
# --------------------------------------------------------------------------- #


def _resolve(root: Path, rel: str) -> Path:
    """Resolve `rel` inside `root`; refuse escapes and secret files."""
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ToolError(f"path escapes the workspace: {rel}")
    if any(part.startswith(".env") for part in path.relative_to(root).parts):
        raise ToolError("access to .env files is blocked")
    if ".git" in path.relative_to(root).parts:
        raise ToolError("access to .git internals is blocked")
    return path


def _clip(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    return text[:MAX_TOOL_OUTPUT] + f"\n...[truncated {len(text) - MAX_TOOL_OUTPUT} chars]"


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise ToolError(f"not a file: {path.name}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ToolError(f"file too large (> {MAX_FILE_BYTES} bytes)")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError("file is not valid UTF-8 text") from exc


async def _t_read_file(root: Path, path: str, start_line: int = 1, end_line: int = 0) -> str:
    lines = _read_text(_resolve(root, path)).splitlines()
    start = max(1, int(start_line))
    end = int(end_line) if end_line else len(lines)
    chunk = lines[start - 1 : end]
    return _clip("\n".join(f"{i}\t{line}" for i, line in enumerate(chunk, start)))


async def _t_list_dir(root: Path, path: str = ".") -> str:
    target = _resolve(root, path)
    if not target.is_dir():
        raise ToolError(f"not a directory: {path}")
    entries = []
    for child in sorted(target.iterdir()):
        if child.name in SKIP_DIRS or child.name.startswith(".env"):
            continue
        entries.append(child.name + ("/" if child.is_dir() else ""))
    return _clip("\n".join(entries) or "(empty)")


async def _t_grep(root: Path, pattern: str, path: str = ".", max_matches: int = 100) -> str:
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise ToolError(f"invalid regex: {exc}") from exc
    base = _resolve(root, path)
    files = [base] if base.is_file() else sorted(base.rglob("*"))
    out: list[str] = []
    for f in files:
        rel = f.relative_to(root)
        if any(p in SKIP_DIRS or p.startswith(".env") for p in rel.parts) or not f.is_file():
            continue
        try:
            if f.stat().st_size > MAX_FILE_BYTES:
                continue
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if rx.search(line):
                    out.append(f"{rel}:{n}: {line[:300]}")
                    if len(out) >= max_matches:
                        return _clip("\n".join(out) + "\n...[max_matches reached]")
        except (UnicodeDecodeError, OSError):
            continue
    return _clip("\n".join(out) or "(no matches)")


async def _t_write_file(root: Path, path: str, content: str) -> str:
    if len(content.encode()) > MAX_FILE_BYTES:
        raise ToolError("content too large")
    target = _resolve(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


async def _t_edit_file(root: Path, path: str, old_string: str, new_string: str) -> str:
    target = _resolve(root, path)
    text = _read_text(target)
    count = text.count(old_string)
    if count != 1:
        raise ToolError(f"old_string must match exactly once (found {count})")
    target.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
    return f"edited {path}"


def _shell_allowed(tokens: list[str]) -> bool:
    return any(tokens[: len(entry)] == entry for entry in SHELL_ALLOWLIST)


async def _t_run_command(root: Path, command: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise ToolError(f"cannot parse command: {exc}") from exc
    if not tokens or not _shell_allowed(tokens):
        allowed = ", ".join(" ".join(e) for e in SHELL_ALLOWLIST)
        raise ToolError(f"command not allowed. Allowed prefixes: {allowed}")
    env = {k: v for k, v in os.environ.items() if k != "OLLAMA_API_KEY"}
    proc = await asyncio.create_subprocess_exec(
        *tokens,
        cwd=root,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=SHELL_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ToolError(f"command timed out after {SHELL_TIMEOUT:.0f}s") from None
    return _clip(f"exit code {proc.returncode}\n" + stdout.decode("utf-8", "replace"))


async def _t_web_search(root: Path, query: str, max_results: int = 5) -> str:
    return _clip(json.dumps(await web_search(query, int(max_results)), ensure_ascii=False))


async def _t_web_fetch(root: Path, url: str) -> str:
    return _clip(json.dumps(await web_fetch(url), ensure_ascii=False))


def _spec(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


_S = {"type": "string"}
_I = {"type": "integer"}

# group -> {tool name: (implementation, schema)}
TOOL_GROUPS: dict[str, dict[str, tuple[Any, dict[str, Any]]]] = {
    "read": {
        "read_file": (
            _t_read_file,
            _spec(
                "read_file",
                "Read a text file in the workspace (numbered lines).",
                {"path": _S, "start_line": _I, "end_line": _I},
                ["path"],
            ),
        ),
        "list_dir": (
            _t_list_dir,
            _spec("list_dir", "List a workspace directory.", {"path": _S}, []),
        ),
        "grep": (
            _t_grep,
            _spec(
                "grep",
                "Regex search over workspace files; returns path:line: text.",
                {"pattern": _S, "path": _S, "max_matches": _I},
                ["pattern"],
            ),
        ),
    },
    "web": {
        "web_search": (
            _t_web_search,
            _spec("web_search", "Search the web.", {"query": _S, "max_results": _I}, ["query"]),
        ),
        "web_fetch": (
            _t_web_fetch,
            _spec("web_fetch", "Fetch a web page's text.", {"url": _S}, ["url"]),
        ),
    },
    "write": {
        "write_file": (
            _t_write_file,
            _spec(
                "write_file",
                "Create or overwrite a text file in the workspace.",
                {"path": _S, "content": _S},
                ["path", "content"],
            ),
        ),
        "edit_file": (
            _t_edit_file,
            _spec(
                "edit_file",
                "Replace one exact, unique occurrence of old_string with new_string.",
                {"path": _S, "old_string": _S, "new_string": _S},
                ["path", "old_string", "new_string"],
            ),
        ),
    },
    "shell": {
        "run_command": (
            _t_run_command,
            _spec(
                "run_command",
                "Run an allow-listed command (no pipes or redirects) in the workspace.",
                {"command": _S},
                ["command"],
            ),
        ),
    },
}


def _group_enabled(group: str) -> bool:
    return {"write": ALLOW_WRITE, "shell": ALLOW_SHELL}.get(group, True)


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #

AGENT_SYSTEM = (
    "You are a sub-agent working for another AI assistant (Claude). Complete the "
    "task using the provided tools. Inspect files before drawing conclusions about "
    "them, and do not guess file contents. When finished, reply WITHOUT calling any "
    "tool, giving a concise final report: what you did, key findings, and any "
    "files you changed. Workspace root is the current directory; use relative paths."
)


async def _execute_tool(
    root: Path, enabled: dict[str, Any], name: str, args: Any
) -> tuple[str, bool]:
    """Run one tool call. Returns (text for the model, ok)."""
    impl = enabled.get(name)
    if impl is None:
        return f"error: unknown or disabled tool '{name}'", False
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return "error: arguments must be a JSON object", False
    if not isinstance(args, dict):
        return "error: arguments must be a JSON object", False
    try:
        return await impl(root, **args), True
    except ToolError as exc:
        return f"error: {exc}", False
    except TypeError as exc:
        return f"error: bad arguments ({exc})", False
    except Exception as exc:  # noqa: BLE001 - never crash the loop on a tool bug
        return f"error: {type(exc).__name__}: {exc}", False


async def _agent_loop(
    client: Any,
    root: Path,
    model: str,
    messages: list[dict[str, Any]],
    enabled: dict[str, Any],
    schemas: list[dict[str, Any]],
    max_steps: int,
    think: Any,
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    for step in range(1, max_steps + 1):
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "tools": schemas}
        if think is not None:
            kwargs["think"] = think
        resp = await client.chat(**kwargs)
        msg = _get(resp, "message", {})
        calls = _get(msg, "tool_calls") or []
        content = _get(msg, "content", "") or ""

        if not calls:
            return {"status": "completed", "content": content, "steps": step}

        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "function": {
                            "name": _get(_get(c, "function"), "name"),
                            "arguments": _get(_get(c, "function"), "arguments") or {},
                        }
                    }
                    for c in calls
                ],
            }
        )
        for c in calls:
            fn = _get(c, "function")
            name, args = _get(fn, "name"), _get(fn, "arguments") or {}
            result, ok = await _execute_tool(root, enabled, name, args)
            trace.append({"step": step, "tool": name, "ok": ok, "args": _brief(args)})
            messages.append({"role": "tool", "tool_name": name, "content": result})

    # Out of steps: ask for a wrap-up without tools.
    messages.append(
        {"role": "user", "content": "Step limit reached. Give your final report now, without tools."}
    )
    resp = await client.chat(model=model, messages=messages)
    content = _get(_get(resp, "message", {}), "content", "") or ""
    return {"status": "max_steps", "content": content, "steps": max_steps}


def _brief(args: Any) -> Any:
    if not isinstance(args, dict):
        return str(args)[:200]
    return {k: (v[:120] + "…" if isinstance(v, str) and len(v) > 120 else v) for k, v in args.items()}


async def agent_run(
    task: str,
    model: str | None = None,
    tools: list[str] | None = None,
    system: str = "",
    max_steps: int = 10,
    timeout_seconds: float | None = None,
    think: bool | Literal["low", "medium", "high"] | None = None,
    workspace: str | None = None,
) -> dict[str, Any]:
    """Run an Ollama model as an autonomous sub-agent and return its final report.

    Args:
        task: Self-contained instructions. The sub-agent sees only this text.
        model: Ollama model; must support tool calling (e.g. gpt-oss:120b, kimi-k3).
        tools: Groups to enable: "read" (read_file/list_dir/grep), "web"
            (web_search/web_fetch), "write" (write_file/edit_file), "shell"
            (run_command). Default ["read", "web"]. "write"/"shell" only work if
            the server operator enabled them in the environment.
        system: Extra role/instructions appended to the built-in agent prompt.
        max_steps: Max model turns (capped by OLLAMA_AGENT_MAX_STEPS).
        timeout_seconds: Wall-clock limit for the whole run.
        think: Optional thinking mode passed to the model.
        workspace: Sub-directory of the workspace root to confine the agent to.
    """
    groups = tools if tools is not None else ["read", "web"]
    unknown = [g for g in groups if g not in TOOL_GROUPS]
    if unknown:
        return {"status": "error", "error": f"unknown tool groups: {unknown}",
                "available": list(TOOL_GROUPS)}
    disabled = [g for g in groups if not _group_enabled(g)]
    if disabled:
        return {
            "status": "error",
            "error": f"tool groups {disabled} are disabled on this server "
            "(set OLLAMA_AGENT_ALLOW_WRITE=1 / OLLAMA_AGENT_ALLOW_SHELL=1 to enable)",
        }

    try:
        root = _resolve(WORKSPACE_ROOT, workspace) if workspace else WORKSPACE_ROOT
    except ToolError as exc:
        return {"status": "error", "error": str(exc)}
    if not root.is_dir():
        return {"status": "error", "error": f"workspace is not a directory: {workspace}"}

    enabled = {n: impl for g in groups for n, (impl, _s) in TOOL_GROUPS[g].items()}
    schemas = [s for g in groups for (_i, s) in TOOL_GROUPS[g].values()]
    used = model or current_model
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM + ("\n\n" + system if system else "")},
        {"role": "user", "content": task},
    ]
    trace: list[dict[str, Any]] = []
    limit = timeout_seconds or AGENT_TIMEOUT
    steps = max(1, min(int(max_steps), MAX_STEPS_CAP))

    async with _agent_slots:
        try:
            result = await asyncio.wait_for(
                _agent_loop(_client(), root, used, messages, enabled, schemas, steps, think, trace),
                timeout=limit,
            )
        except asyncio.TimeoutError:
            result = {"status": "timeout", "content": "", "steps": len(trace),
                      "error": f"agent exceeded {limit:.0f}s"}
        except Exception as exc:  # noqa: BLE001
            result = {"status": "error", "content": "", "steps": len(trace),
                      "error": f"{type(exc).__name__}: {exc}"}

    return {"model": used, "tools": groups, "tool_calls": trace, **result}


def agent_tools() -> dict[str, Any]:
    """Describe agent tool groups and which are enabled on this server."""
    return {
        "workspace_root": str(WORKSPACE_ROOT),
        "max_concurrent_agents": MAX_CONCURRENT_AGENTS,
        "groups": {
            g: {"enabled": _group_enabled(g), "tools": list(t)} for g, t in TOOL_GROUPS.items()
        },
        "shell_allowlist": [" ".join(e) for e in SHELL_ALLOWLIST] if ALLOW_SHELL else [],
    }


for _fn in (
    list_models,
    get_current_model,
    set_current_model,
    chat,
    web_search,
    web_fetch,
    agent_run,
    agent_tools,
):
    mcp.tool(_fn)


if __name__ == "__main__":
    mcp.run()

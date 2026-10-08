"""Offline tests for agent_run (no network, scripted fake Ollama client)."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server  # noqa: E402


def call(name, **args):
    return NS(function=NS(name=name, arguments=args))


def reply(content="", calls=None):
    return NS(message=NS(content=content, tool_calls=calls or None, thinking=None))


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    async def chat(self, **kwargs):
        self.seen.append(kwargs)
        item = self.script.pop(0)
        if callable(item):
            return await item()
        return item


@pytest.fixture
def ws(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("print('hi')\n# TODO fix\n")
    (tmp_path / ".env").write_text("OLLAMA_API_KEY=secret\n")
    monkeypatch.setattr(server, "WORKSPACE_ROOT", tmp_path.resolve())
    return tmp_path.resolve()


def use(monkeypatch, client):
    monkeypatch.setattr(server, "_client", lambda: client)


async def test_read_tools_and_final_report(ws, monkeypatch):
    client = FakeClient(
        [
            reply(calls=[call("grep", pattern="TODO"), call("read_file", path="a.py")]),
            reply("Found one TODO in a.py:2"),
        ]
    )
    use(monkeypatch, client)
    out = await server.agent_run("find todos", model="m")
    assert out["status"] == "completed" and "TODO" in out["content"]
    assert [t["tool"] for t in out["tool_calls"]] == ["grep", "read_file"]
    tool_msgs = [m for m in client.seen[-1]["messages"] if m["role"] == "tool"]
    assert "a.py:2" in tool_msgs[0]["content"]
    assert "1\tprint('hi')" in tool_msgs[1]["content"]


async def test_path_escape_and_env_blocked(ws, monkeypatch):
    client = FakeClient(
        [
            reply(calls=[call("read_file", path="../etc/passwd"), call("read_file", path=".env")]),
            reply("done"),
        ]
    )
    use(monkeypatch, client)
    out = await server.agent_run("x", model="m")
    assert [t["ok"] for t in out["tool_calls"]] == [False, False]
    results = [m["content"] for m in client.seen[-1]["messages"] if m["role"] == "tool"]
    assert "escapes" in results[0] and ".env" in results[1]
    assert "secret" not in "".join(results)


async def test_write_requires_server_and_call_opt_in(ws, monkeypatch):
    use(monkeypatch, FakeClient([]))
    out = await server.agent_run("x", tools=["write"])
    assert out["status"] == "error" and "disabled" in out["error"]

    monkeypatch.setattr(server, "ALLOW_WRITE", True)
    client = FakeClient(
        [
            reply(calls=[call("write_file", path="sub/n.txt", content="hello")]),
            reply(calls=[call("edit_file", path="sub/n.txt", old_string="hello", new_string="bye")]),
            reply("ok"),
        ]
    )
    use(monkeypatch, client)
    out = await server.agent_run("x", model="m", tools=["read", "write"])
    assert out["status"] == "completed"
    assert (ws / "sub" / "n.txt").read_text() == "bye"


async def test_disabled_tool_not_callable(ws, monkeypatch):
    client = FakeClient([reply(calls=[call("write_file", path="x", content="y")]), reply("ok")])
    use(monkeypatch, client)
    out = await server.agent_run("x", model="m")  # default read+web only
    assert out["tool_calls"][0]["ok"] is False
    assert not (ws / "x").exists()


async def test_shell_allowlist(ws, monkeypatch):
    monkeypatch.setattr(server, "ALLOW_SHELL", True)
    client = FakeClient(
        [
            reply(
                calls=[
                    call("run_command", command="cat a.py"),
                    call("run_command", command="rm -rf /"),
                    call("run_command", command="cat a.py; rm a.py"),
                ]
            ),
            reply("ok"),
        ]
    )
    use(monkeypatch, client)
    out = await server.agent_run("x", model="m", tools=["shell"])
    assert [t["ok"] for t in out["tool_calls"]] == [True, False, True]
    # ';' is a literal argument to cat (no shell), so nothing was deleted
    assert (ws / "a.py").exists()


async def test_max_steps_forces_wrapup(ws, monkeypatch):
    loop = [reply(calls=[call("list_dir")]) for _ in range(2)]
    client = FakeClient(loop + [reply("wrap-up")])
    use(monkeypatch, client)
    out = await server.agent_run("x", model="m", max_steps=2)
    assert out["status"] == "max_steps" and out["content"] == "wrap-up"


async def test_timeout_returns_partial(ws, monkeypatch):
    async def hang():
        await asyncio.sleep(5)

    use(monkeypatch, FakeClient([hang]))
    out = await server.agent_run("x", model="m", timeout_seconds=0.2)
    assert out["status"] == "timeout"


async def test_model_error_is_reported_not_raised(ws, monkeypatch):
    async def boom():
        raise RuntimeError("model does not support tools")

    use(monkeypatch, FakeClient([boom]))
    out = await server.agent_run("x", model="m")
    assert out["status"] == "error" and "support tools" in out["error"]


async def test_parallel_agents_respect_semaphore(ws, monkeypatch):
    active = peak = 0

    async def slow():
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return reply("done")

    monkeypatch.setattr(server, "_agent_slots", asyncio.Semaphore(2))
    monkeypatch.setattr(server, "_client", lambda: FakeClient([slow]))
    outs = await asyncio.gather(*[server.agent_run("x", model="m") for _ in range(5)])
    assert all(o["status"] == "completed" for o in outs)
    assert peak == 2


async def test_workspace_subdir_confined(ws, monkeypatch):
    (ws / "pkg").mkdir()
    (ws / "pkg" / "b.py").write_text("x = 1\n")
    client = FakeClient([reply(calls=[call("read_file", path="../a.py")]), reply("ok")])
    use(monkeypatch, client)
    out = await server.agent_run("x", model="m", workspace="pkg")
    assert out["tool_calls"][0]["ok"] is False
    bad = await server.agent_run("x", model="m", workspace="../")
    assert bad["status"] == "error"


async def test_mcp_registration():
    names = {t.name for t in await server.mcp.list_tools()}
    assert {"agent_run", "agent_tools", "chat", "list_models", "web_search"} <= names

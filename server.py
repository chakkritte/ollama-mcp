import os
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from ollama import Client

load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "https://ollama.com")
ollama_api_key = os.getenv("OLLAMA_API_KEY", "")

client = Client(
    host=ollama_host,
    headers={
        "Authorization": f"Bearer {ollama_api_key}"
    },
)

current_model = os.getenv("DEFAULT_MODEL")

mcp = FastMCP("Ollama Cloud")


@mcp.tool()
def list_models():
    """List all available Ollama Cloud models."""
    return client.list()


@mcp.tool()
def get_current_model():
    """Return the current default model."""
    return current_model


@mcp.tool()
def set_current_model(model: str):
    """Change the default model used for chat."""
    global current_model
    current_model = model
    return {
        "success": True,
        "current_model": current_model,
    }


@mcp.tool()
def chat(
    prompt: str,
    model: str | None = None,
    system: str = "",
    think: bool | Literal["low", "medium", "high"] | None = None,
):
    """Chat with an Ollama Cloud model.

    Set `think` to enable extended thinking/reasoning mode:
    - `True`, or one of `"low"`, `"medium"`, `"high"` to control the thinking effort
      (model support varies — e.g. `deepseek-v4-pro`, `glm-5.2`, `kimi-k2.5`).
    - `False` or `None` to disable thinking.

    When thinking is enabled, the response includes a `thinking` field with the
    model's reasoning trace, alongside the final `content`.
    """

    use_model = model or current_model

    messages = []

    if system:
        messages.append(
            {
                "role": "system",
                "content": system,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    response = client.chat(
        model=use_model,
        messages=messages,
        think=think,
    )

    message = response["message"]

    return {
        "model": use_model,
        "content": message["content"],
        "thinking": message["thinking"] or None,
    }


@mcp.tool()
def web_search(query: str, max_results: int = 5):
    """Search the web using Ollama's hosted web search API.

    Returns up to `max_results` results (capped at 10), each with a title,
    URL, and a short snippet of content.
    """
    max_results = min(max(1, max_results), 10)

    try:
        with httpx.Client(timeout=30.0) as http:
            resp = http.post(
                f"{ollama_host}/api/web_search",
                headers={
                    "Authorization": f"Bearer {ollama_api_key}",
                    "Content-Type": "application/json",
                },
                json={"query": query, "max_results": max_results},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        return {
            "error": f"HTTP {e.response.status_code} from Ollama web_search API",
            "detail": e.response.text,
        }
    except httpx.RequestError as e:
        return {"error": "Request error", "detail": str(e)}

    results = data.get("results", [])
    if not results:
        return {"query": query, "results": []}

    return {
        "query": query,
        "results": [
            {
                "title": r.get("title", "Untitled"),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in results[:max_results]
        ],
    }


@mcp.tool()
def web_fetch(url: str):
    """Fetch a webpage using Ollama's hosted web fetch API.

    Returns the page title, the main text content, and up to 10 links
    found on the page.
    """
    try:
        with httpx.Client(timeout=30.0) as http:
            resp = http.post(
                f"{ollama_host}/api/web_fetch",
                headers={
                    "Authorization": f"Bearer {ollama_api_key}",
                    "Content-Type": "application/json",
                },
                json={"url": url},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        return {
            "error": f"HTTP {e.response.status_code} from Ollama web_fetch API",
            "detail": e.response.text,
        }
    except httpx.RequestError as e:
        return {"error": "Request error", "detail": str(e)}

    return {
        "url": url,
        "title": data.get("title", "Untitled"),
        "content": data.get("content", ""),
        "links": data.get("links", [])[:10],
    }


if __name__ == "__main__":
    mcp.run()
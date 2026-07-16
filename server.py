import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from ollama import Client

load_dotenv()

client = Client(
    host=os.getenv("OLLAMA_HOST"),
    headers={
        "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
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
):
    """Chat with an Ollama Cloud model."""

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
    )

    return {
        "model": use_model,
        "content": response["message"]["content"],
    }


if __name__ == "__main__":
    mcp.run()
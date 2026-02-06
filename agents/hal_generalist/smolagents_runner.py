"""Core smolagents runner — mirrors HAL's hal_generalist_agent/main.py."""

import os

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    WebSearchTool,
    VisitWebpageTool,
)
from .tools import read_file_tool
from .prompts import GAIA_SYSTEM_PROMPT


def run_gaia_task(
    task_id: str,
    task_data: dict,
    model_id: str = "openrouter/deepseek/deepseek-chat-v3-0324",
    max_steps: int = 15,
) -> str:
    """Run a single GAIA task using smolagents CodeAgent.

    Args:
        task_id: Task identifier.
        task_data: Dict with keys like "question", "file_path", "file_name".
        model_id: LiteLLM-compatible model string.
        max_steps: Maximum agent reasoning steps.

    Returns:
        The agent's final answer as a string.
    """

    # Model — uses LiteLLMModel which routes through existing
    # litellm/openrouter setup. No separate deepseek provider needed.
    effective_model = model_id or os.environ.get("HARNESS_MODEL", "openrouter/deepseek/deepseek-chat-v3-0324")
    model = LiteLLMModel(model_id=effective_model)

    # Tools — same as HAL generalist
    tools = [
        WebSearchTool(),       # DuckDuckGo search (free, no API key)
        VisitWebpageTool(),    # Fetch + parse webpage content
        read_file_tool,        # Custom: read GAIA file attachments
    ]

    # Build agent
    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=max_steps,
        additional_authorized_imports=[
            "json", "csv", "re", "math", "datetime",
            "collections", "itertools", "functools",
            "os", "pathlib", "io", "base64",
            "openpyxl", "pymupdf",
        ],
    )

    # Build the task prompt
    question = task_data.get("Question") or task_data.get("question", "")
    file_name = task_data.get("file_name", "")
    file_path = task_data.get("file_path", "")

    prompt = GAIA_SYSTEM_PROMPT + "\n\n" + question
    if file_name and file_path:
        prompt += f"\n\nA file is provided: {file_name} (path: {file_path})"
        prompt += "\nYou can read it using the read_file tool."

    # Run
    try:
        result = agent.run(prompt)
        return str(result).strip()
    except Exception as e:
        return f"ERROR: {e}"

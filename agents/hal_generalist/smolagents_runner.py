"""Core smolagents runner — mirrors HAL's hal_generalist_agent/main.py."""

import json
import os
import sys
import time
from typing import Callable

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    WebSearchTool,
    VisitWebpageTool,
)
from .tools import read_file_tool
from .prompts import GAIA_SYSTEM_PROMPT


def _emit_log_fallback(data: dict) -> None:
    """Emit a JSON-RPC log message on stdout (fallback when no callback)."""
    if "timestamp" not in data:
        data["timestamp"] = time.time()
    msg = {"jsonrpc": "2.0", "method": "log", "params": data}
    print(json.dumps(msg), flush=True)


def _make_step_callback(
    model_id: str = "unknown",
    completion_callback: Callable | None = None,
):
    """Create a step callback closure.

    If *completion_callback* is provided (``Agent.emit_completion``), it is
    used to report usage through the proper harness contract.  Otherwise we
    fall back to emitting raw JSON-RPC on stdout — still captured by the
    runner, but bypasses the Agent warning system.
    """

    def _step_callback(step):
        step_num = getattr(step, "step_number", "?")
        error = getattr(step, "error", None)
        code = getattr(step, "code_action", None)
        obs = getattr(step, "observations", None)
        is_final = getattr(step, "is_final_answer", False)
        token_usage = getattr(step, "token_usage", None)

        # Human-readable progress on stderr
        parts = [f"[step {step_num}]"]
        if is_final:
            parts.append("FINAL")
        if error:
            parts.append(f"error={error}")
        if code:
            parts.append(f"code={code[:120]}...")
        if obs:
            parts.append(f"obs={str(obs)[:120]}...")
        if token_usage:
            parts.append(f"tokens={token_usage}")
        print(" ".join(parts), file=sys.stderr, flush=True)

        # Structured completion event for harness usage tracking
        if token_usage:
            input_tokens = getattr(token_usage, "input_tokens", 0) or 0
            output_tokens = getattr(token_usage, "output_tokens", 0) or 0

            if completion_callback is not None:
                # Preferred: use the Agent.emit_completion() contract
                completion_callback(
                    model=model_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    provider="litellm/openrouter",
                )
            else:
                # Fallback: raw JSON-RPC on stdout
                total_tokens = getattr(token_usage, "total_tokens", 0) or 0
                _emit_log_fallback({
                    "type": "completion",
                    "provider": "litellm/openrouter",
                    "request": {"model": model_id},
                    "response": {
                        "model": model_id,
                        "usage": {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": total_tokens or (input_tokens + output_tokens),
                        },
                    },
                    "latency_ms": 0,
                    "step": step_num,
                })

    return _step_callback


def run_gaia_task(
    task_id: str,
    task_data: dict,
    model_id: str = "openrouter/deepseek/deepseek-chat-v3-0324",
    max_steps: int = 15,
    completion_callback: Callable | None = None,
) -> str:
    """Run a single GAIA task using smolagents CodeAgent.

    Args:
        task_id: Task identifier.
        task_data: Dict with keys like "question", "file_path", "file_name".
        model_id: LiteLLM-compatible model string.
        max_steps: Maximum agent reasoning steps.
        completion_callback: Optional callback (``Agent.emit_completion``)
            for reporting per-step usage.  When *None*, falls back to raw
            JSON-RPC on stdout.

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

    # Build step callback with the proper completion reporting path
    step_cb = _make_step_callback(
        model_id=effective_model,
        completion_callback=completion_callback,
    )

    # Build agent
    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=max_steps,
        step_callbacks=[step_cb],
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

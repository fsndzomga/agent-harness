"""Core smolagents runner for GAIA Agent v1."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from typing import Callable

from smolagents import CodeAgent, LiteLLMModel, WebSearchTool, VisitWebpageTool

from .compaction import StepCompactor
from .prompts import GAIA_SYSTEM_PROMPT
from .tools import (
    read_file_tool,
    save_note,
    read_notes,
    wiki_search,
    set_memory_dir,
)


# ------------------------------------------------------------------
# Step callback factory
# ------------------------------------------------------------------

def _make_step_callback(
    model_id: str,
    completion_callback: Callable | None,
    compactor: StepCompactor | None,
    agent_ref_holder: list,           # mutable slot [agent]
):
    """Build the step callback that:
    1. Emits completion events (usage tracking).
    2. Triggers context compaction.
    """

    def _callback(step):
        step_num = getattr(step, "step_number", "?")
        error = getattr(step, "error", None)
        code = getattr(step, "code_action", None)
        obs = getattr(step, "observations", None)
        is_final = getattr(step, "is_final_answer", False)
        token_usage = getattr(step, "token_usage", None)

        # --- stderr progress ---
        parts = [f"[step {step_num}]"]
        if is_final:
            parts.append("FINAL")
        if error:
            parts.append(f"error={error}")
        if code:
            parts.append(f"code={code[:120]}…")
        if obs:
            parts.append(f"obs_len={len(str(obs))}")
        if token_usage:
            parts.append(f"tokens={token_usage}")
        print(" ".join(parts), file=sys.stderr, flush=True)

        # --- completion event ---
        if token_usage:
            input_tokens = getattr(token_usage, "input_tokens", 0) or 0
            output_tokens = getattr(token_usage, "output_tokens", 0) or 0

            if completion_callback is not None:
                completion_callback(
                    model=model_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    provider="litellm/openrouter",
                )

        # --- context compaction ---
        if compactor and agent_ref_holder:
            compactor.compact(agent_ref_holder[0])

    return _callback


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def run_gaia_task(
    task_id: str,
    task_data: dict,
    model_id: str = "openrouter/deepseek/deepseek-chat-v3-0324",
    max_steps: int = 60,
    completion_callback: Callable | None = None,
) -> str:
    """Run a single GAIA task with the v1 agent.

    Args:
        task_id: Unique task identifier.
        task_data: Dict with "question", optional "file_path" / "file_name".
        model_id: LiteLLM model string.
        max_steps: Maximum reasoning steps.
        completion_callback: ``Agent.emit_completion`` for usage tracking.

    Returns:
        The agent's final answer (string).
    """

    effective_model = model_id or os.environ.get(
        "HARNESS_MODEL", "openrouter/deepseek/deepseek-chat-v3-0324"
    )

    # --- per-task memory directory ---
    mem_dir = tempfile.mkdtemp(prefix=f"gaia_mem_{task_id[:12]}_")
    set_memory_dir(mem_dir)

    # --- model ---
    model = LiteLLMModel(model_id=effective_model)

    # --- tools ---
    tools = [
        WebSearchTool(),       # DuckDuckGo search (smolagents built-in)
        VisitWebpageTool(),    # Fetch + parse webpage (smolagents built-in)
        wiki_search,           # Wikipedia API lookup
        read_file_tool,        # Enhanced file reader (PDF, Excel, audio, images, etc.)
        save_note,             # Persistent memory: save findings
        read_notes,            # Persistent memory: read all notes
    ]

    # --- compactor ---
    compactor = StepCompactor(
        memory_dir=mem_dir,
        keep_recent=20,
        max_obs_chars=600,
    )

    # --- agent (mutable holder so callback can reference it) ---
    agent_ref: list = []

    step_cb = _make_step_callback(
        model_id=effective_model,
        completion_callback=completion_callback,
        compactor=compactor,
        agent_ref_holder=agent_ref,
    )

    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=max_steps,
        step_callbacks=[step_cb],
        additional_authorized_imports=[
            "json", "csv", "re", "math", "datetime", "time",
            "collections", "itertools", "functools",
            "os", "pathlib", "io", "base64", "struct",
            "openpyxl", "pymupdf",
            "statistics", "fractions", "decimal",
            "hashlib", "urllib", "string",
        ],
    )
    agent_ref.append(agent)

    # --- prompt ---
    question = task_data.get("Question") or task_data.get("question", "")
    file_name = task_data.get("file_name", "")
    file_path = task_data.get("file_path", "")

    prompt = GAIA_SYSTEM_PROMPT + "\n\n" + question
    if file_name and file_path:
        prompt += f"\n\nAn attached file is provided: **{file_name}** (path: `{file_path}`)"
        prompt += "\nRead it with read_file_tool(file_path) to inspect the contents."

    # --- run ---
    try:
        result = agent.run(prompt)
        return str(result).strip()
    except Exception as e:
        return f"ERROR: {e}"

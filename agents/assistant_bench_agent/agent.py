#!/usr/bin/env python3
"""AssistantBench Memory Agent — a minimal, memory-driven research agent.

Architecture:
  1. A markdown memory file tracks every action and result.
  2. Each turn, the LLM sees the full memory and outputs one action.
  3. The agent parses the action, executes it, appends the result to memory.
  4. Loop until [ANSWER] or max steps reached.

The memory file is the single source of truth — the LLM has no hidden state
beyond what is recorded there.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from datetime import datetime, timezone

# Ensure the agents/ directory is on sys.path for sibling imports
_agent_dir = os.path.dirname(os.path.abspath(__file__))
_agents_dir = os.path.dirname(_agent_dir)
if _agents_dir not in sys.path:
    sys.path.insert(0, _agents_dir)

from harness.agent import Agent
from harness.providers.base import Message
from assistant_bench_agent.prompts import SYSTEM_PROMPT
from assistant_bench_agent.tools import web_search, visit_webpage

# ── defaults ──────────────────────────────────────────────────────────

MAX_STEPS = 15
MAX_MEMORY_CHARS = 48_000  # truncate old memory if it gets too large


class AssistantBenchAgent(Agent):
    """Minimal memory-based agent for AssistantBench tasks."""

    def run_task(self, task_id: str, task_data: dict) -> str:
        question = task_data.get("question") or task_data.get("task", "")
        max_steps = int(os.environ.get("AGENT_MAX_STEPS", MAX_STEPS))

        # ── Initialise memory file ────────────────────────────────────
        memory_dir = tempfile.mkdtemp(prefix="ab_memory_")
        memory_path = os.path.join(memory_dir, f"{task_id}.md")

        self._init_memory(memory_path, task_id, question)
        self.set_metadata("memory_path", memory_path)
        self.set_metadata("max_steps", max_steps)

        self.log("agent_start", task_id=task_id, memory_path=memory_path)

        answer = ""
        for step in range(1, max_steps + 1):
            self.increment("steps")

            # Read current memory
            memory = self._read_memory(memory_path)

            # Ask LLM what to do next
            response = self.complete([
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=(
                    f"## Your Memory (read carefully)\n\n{memory}\n\n"
                    "---\n"
                    "Now output your next action."
                )),
            ], max_tokens=2048, temperature=0.0)

            raw_output = response.message.content.strip()
            self.log("llm_output", step=step, output=raw_output[:500])

            # Parse action from LLM output
            action, payload = self._parse_action(raw_output)
            self.record_tool_use(action)

            if action == "ANSWER":
                answer = payload
                self._append_memory(memory_path, step, "Final Answer", answer)
                self.log("answer", step=step, answer=answer)
                break

            elif action == "SEARCH":
                self.log("search", step=step, query=payload)
                results = web_search(payload)
                self._append_memory(
                    memory_path, step,
                    f"Web Search — query: `{payload}`",
                    results,
                )

            elif action == "VISIT":
                self.log("visit", step=step, url=payload)
                content = visit_webpage(payload)
                self._append_memory(
                    memory_path, step,
                    f"Visit Page — {payload}",
                    content,
                )

            elif action == "THINK":
                self.log("think", step=step, thought=payload[:300])
                self._append_memory(
                    memory_path, step,
                    "Reasoning",
                    payload,
                )

            else:
                # Unknown action — record it and continue
                self.log("unknown_action", step=step, raw=raw_output[:300])
                self._append_memory(
                    memory_path, step,
                    "Unparsed Output",
                    raw_output,
                )

        # If we ran out of steps without [ANSWER], extract one from memory
        if not answer:
            answer = self._fallback_answer(memory_path, question)
            self.log("fallback_answer", answer=answer)

        self.metric("final_answer", answer)
        return answer

    # ── Memory management ─────────────────────────────────────────────

    def _init_memory(self, path: str, task_id: str, question: str) -> None:
        """Create the initial memory file."""
        header = (
            f"# Memory — Task {task_id}\n\n"
            f"**Started:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"## Task\n\n{question}\n\n"
            f"## Research Log\n"
        )
        with open(path, "w") as f:
            f.write(header)

    def _read_memory(self, path: str) -> str:
        """Read the full memory, truncating the middle if too large."""
        with open(path) as f:
            content = f.read()

        if len(content) <= MAX_MEMORY_CHARS:
            return content

        # Keep the beginning (task + first steps) and end (recent steps)
        keep_start = MAX_MEMORY_CHARS // 3
        keep_end = MAX_MEMORY_CHARS * 2 // 3
        return (
            content[:keep_start]
            + "\n\n... (earlier research truncated for brevity) ...\n\n"
            + content[-keep_end:]
        )

    def _append_memory(
        self, path: str, step: int, heading: str, content: str
    ) -> None:
        """Append a step entry to the memory file."""
        entry = f"\n### Step {step}: {heading}\n\n{content}\n"
        with open(path, "a") as f:
            f.write(entry)

    # ── Action parsing ────────────────────────────────────────────────

    def _parse_action(self, text: str) -> tuple[str, str]:
        """Parse an action block from the LLM's output.

        Returns (action_type, payload).
        Action types: SEARCH, VISIT, THINK, ANSWER, UNKNOWN.
        """
        # Try to match [ACTION] payload patterns
        action_map = {
            "ANSWER": r"\[ANSWER\]\s*(.*)",
            "SEARCH": r"\[SEARCH\]\s*(.*)",
            "VISIT":  r"\[VISIT\]\s*(.*)",
            "THINK":  r"\[THINK\]\s*(.*)",
        }
        for action_name, pattern in action_map.items():
            m = re.search(pattern, text, re.DOTALL)
            if m:
                payload = m.group(1).strip()
                # Clean markdown artifacts from answers
                if action_name == "ANSWER":
                    payload = self._clean_answer(payload)
                return action_name, payload

        # Fallback: if the whole thing looks like an answer (short, no action tag)
        if len(text) < 200 and not any(
            kw in text.upper() for kw in ["SEARCH", "VISIT", "THINK"]
        ):
            return "ANSWER", text

        return "UNKNOWN", text

    @staticmethod
    def _clean_answer(text: str) -> str:
        """Strip markdown artifacts and noise from an answer."""
        # Remove code block fences
        text = re.sub(r"```\w*\n?", "", text)
        # Remove leading/trailing quotes
        text = text.strip().strip("'\"").strip()
        # Remove common LLM preamble patterns
        text = re.sub(
            r"^(Based on .*?research[,.]?\s*|According to .*?[,.]?\s*|"
            r"From .*?findings[,.]?\s*|The answer is[:\s]*)",
            "", text, flags=re.IGNORECASE,
        )
        return text.strip()

    # ── Fallback answer extraction ────────────────────────────────────

    def _fallback_answer(self, memory_path: str, question: str) -> str:
        """If the agent ran out of steps, ask the LLM to synthesize from memory."""
        memory = self._read_memory(memory_path)

        self.increment("steps")
        response = self.complete([
            Message(role="system", content=(
                "You are given a research memory document and a question. "
                "Based ONLY on the information in the memory, give a precise, "
                "concise answer. Output ONLY the answer — no explanation."
            )),
            Message(role="user", content=(
                f"## Question\n{question}\n\n"
                f"## Research Memory\n{memory}"
            )),
        ], max_tokens=1024, temperature=0.0)

        answer = response.message.content.strip()
        answer = self._clean_answer(answer)
        self._append_memory(memory_path, 999, "Fallback Answer (out of steps)", answer)
        return answer


if __name__ == "__main__":
    AssistantBenchAgent().run()

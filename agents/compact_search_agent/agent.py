#!/usr/bin/env python3
"""Compact Search Agent — keyword search with LLM-driven memory compaction.

Architecture:
  1. A markdown memory file tracks every action and result.
  2. Each turn, the LLM sees the full memory and outputs one action.
  3. The agent parses the action, executes it, appends the result to memory.
  4. When memory exceeds a threshold, older entries are **compacted** —
     summarised by the same LLM model into a dense bullet-point recap.
  5. Loop until [ANSWER] or max steps reached.

The key difference from the base AssistantBench agent is that compaction
replaces the naive middle-truncation with an intelligent LLM-driven summary,
preserving all critical facts while reclaiming context space.
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
from compact_search_agent.prompts import SYSTEM_PROMPT, COMPACTION_PROMPT
from compact_search_agent.tools import web_search, visit_webpage

# ── defaults ──────────────────────────────────────────────────────────

MAX_STEPS = 15
COMPACT_THRESHOLD = 24_000   # compact when raw memory exceeds this (chars)
COMPACT_TARGET = 12_000      # target size for the compacted summary
MAX_MEMORY_CHARS = 48_000    # hard cap for safety (after compaction)
KEEP_RECENT_STEPS = 4        # always keep the last N steps verbatim
MAX_DEGEN_RETRIES = 2        # retries when LLM produces degenerate output


class CompactSearchAgent(Agent):
    """Keyword-search agent with LLM-driven memory compaction."""

    def run_task(self, task_id: str, task_data: dict) -> str:
        question = task_data.get("question") or task_data.get("task", "")
        max_steps = int(os.environ.get("AGENT_MAX_STEPS", MAX_STEPS))

        # ── Initialise memory file ────────────────────────────────────
        memory_dir = tempfile.mkdtemp(prefix="cs_memory_")
        memory_path = os.path.join(memory_dir, f"{task_id}.md")

        self._init_memory(memory_path, task_id, question)
        self.set_metadata("memory_path", memory_path)
        self.set_metadata("max_steps", max_steps)
        self.set_metadata("agent_variant", "compact_search_agent")

        self.log("agent_start", task_id=task_id, memory_path=memory_path)

        answer = ""
        for step in range(1, max_steps + 1):
            self.increment("steps")

            # ── Compaction check ──────────────────────────────────────
            self._maybe_compact(memory_path, step)

            # Read current memory
            memory = self._read_memory(memory_path)

            # Ask LLM what to do next (with degeneration retry)
            raw_output = None
            for attempt in range(1 + MAX_DEGEN_RETRIES):
                temp = 0.0 if attempt == 0 else 0.3 + 0.2 * attempt
                response = self.complete([
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=(
                        f"## Your Memory (read carefully)\n\n{memory}\n\n"
                        "---\n"
                        "Now output your next action."
                    )),
                ], max_tokens=2048, temperature=temp)

                raw_output = response.message.content.strip()

                if not self._is_degenerate(raw_output):
                    break
                self.log("degenerate_retry", step=step, attempt=attempt,
                         output=raw_output[:200])
                self.increment("degenerate_retries")

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
        """Read the full memory, with a hard-cap safety truncation."""
        with open(path) as f:
            content = f.read()

        # Hard cap — should rarely trigger after compaction
        if len(content) <= MAX_MEMORY_CHARS:
            return content

        keep_start = MAX_MEMORY_CHARS // 3
        keep_end = MAX_MEMORY_CHARS * 2 // 3
        return (
            content[:keep_start]
            + "\n\n... (emergency truncation — compaction should prevent this) ...\n\n"
            + content[-keep_end:]
        )

    def _read_raw_memory(self, path: str) -> str:
        """Read the raw memory file without any truncation."""
        with open(path) as f:
            return f.read()

    def _append_memory(
        self, path: str, step: int, heading: str, content: str
    ) -> None:
        """Append a step entry to the memory file."""
        entry = f"\n### Step {step}: {heading}\n\n{content}\n"
        with open(path, "a") as f:
            f.write(entry)

    # ── Compaction ────────────────────────────────────────────────────

    def _maybe_compact(self, memory_path: str, current_step: int) -> None:
        """If memory exceeds the threshold, compact older entries via LLM.

        Splits the memory into:
          - header (task description) — always kept
          - old steps — summarised by the LLM into a compact recap
          - recent steps (last KEEP_RECENT_STEPS) — kept verbatim

        The compacted summary replaces the old steps in the memory file.
        """
        raw = self._read_raw_memory(memory_path)
        if len(raw) <= COMPACT_THRESHOLD:
            return  # Nothing to do

        self.log("compaction_triggered", step=current_step, memory_chars=len(raw))

        # ── Split memory into sections ────────────────────────────────
        header, steps = self._split_header_and_steps(raw)

        if len(steps) <= KEEP_RECENT_STEPS:
            return  # Not enough steps to compact

        old_steps = steps[:-KEEP_RECENT_STEPS]
        recent_steps = steps[-KEEP_RECENT_STEPS:]

        # Build the text of old steps to summarise
        old_text = "\n".join(old_steps)

        # Check if there's already a compacted summary — include it for
        # continuity so the LLM can merge old + new compacted content.
        existing_summary = ""
        if "## Compacted Summary" in header:
            parts = header.split("## Research Log", 1)
            if len(parts) == 2:
                # The header already contains a compacted summary section
                existing_summary = "(Previous compacted summary is included in the header above.)\n\n"

        # ── Ask the LLM to summarise ─────────────────────────────────
        self.increment("compaction_calls")
        compaction_prompt = COMPACTION_PROMPT.format(max_chars=COMPACT_TARGET)

        response = self.complete([
            Message(role="system", content=compaction_prompt),
            Message(role="user", content=(
                f"{existing_summary}"
                f"## Notes to condense\n\n{old_text}"
            )),
        ], max_tokens=2048, temperature=0.0)

        summary = response.message.content.strip()
        self.log(
            "compaction_done",
            old_chars=len(old_text),
            summary_chars=len(summary),
        )

        # ── Rebuild the memory file ───────────────────────────────────
        # Extract the pure header (up to "## Research Log")
        pure_header = header.split("## Compacted Summary")[0].rstrip()
        if not pure_header.endswith("## Research Log"):
            # Find "## Research Log" position
            log_pos = pure_header.find("## Research Log")
            if log_pos >= 0:
                pure_header = pure_header[:log_pos + len("## Research Log")]
            else:
                pure_header = pure_header + "\n\n## Research Log"

        new_memory = (
            f"{pure_header}\n\n"
            f"## Compacted Summary (steps 1–{current_step - KEEP_RECENT_STEPS})\n\n"
            f"{summary}\n\n"
            f"## Recent Steps\n"
            + "\n".join(recent_steps)
        )

        with open(memory_path, "w") as f:
            f.write(new_memory)

        self.metric("last_compaction_step", current_step)
        self.metric("compacted_memory_chars", len(new_memory))

    @staticmethod
    def _split_header_and_steps(raw: str) -> tuple[str, list[str]]:
        """Split raw memory into the header and a list of step blocks.

        Each step block starts with '### Step'.
        Returns (header_text, [step_block_1, step_block_2, ...]).
        """
        # Split on step headings, keeping the delimiter
        parts = re.split(r"(?=\n### Step )", raw)

        header = parts[0]
        steps = [p for p in parts[1:] if p.strip()]

        return header, steps

    # ── Degeneration detection ─────────────────────────────────────────

    @staticmethod
    def _is_degenerate(text: str) -> bool:
        """Detect LLM token degeneration (repetitive garbage output)."""
        if not text or len(text) < 3:
            return True
        # Strip whitespace for analysis
        chars = text.replace("\n", "").replace(" ", "").replace("\t", "")
        if not chars:
            return True
        # Very low character diversity → repetitive loop
        if len(set(chars)) / max(len(chars), 1) < 0.10:
            return True
        # No recognisable words (at least 3-letter sequences) in first 150 chars
        if not re.search(r"[a-zA-Z]{3,}", text[:150]):
            return True
        # Detect token-level repetition (same 3-8 char chunk repeated 5+ times)
        if re.search(r"(.{3,8})\1{4,}", chars):
            return True
        return False

    # ── Action parsing ────────────────────────────────────────────────

    def _parse_action(self, text: str) -> tuple[str, str]:
        """Parse an action block from the LLM's output.

        Returns (action_type, payload).
        Action types: SEARCH, VISIT, THINK, ANSWER, UNKNOWN.
        """
        # If the output is degenerate, never accept it as an answer
        if self._is_degenerate(text):
            return "UNKNOWN", text

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
                if action_name == "ANSWER":
                    payload = self._clean_answer(payload)
                return action_name, payload

        # Fallback: only treat as answer if it looks like a real answer
        # (short, contains real words, no action keywords)
        if (
            len(text) < 200
            and re.search(r"[a-zA-Z]{4,}", text)
            and not any(kw in text.upper() for kw in ["SEARCH", "VISIT", "THINK"])
        ):
            return "ANSWER", self._clean_answer(text)

        return "UNKNOWN", text

    @staticmethod
    def _clean_answer(text: str) -> str:
        """Strip markdown artifacts, noise, and excess detail from an answer."""
        # Remove markdown code fences
        text = re.sub(r"```\w*\n?", "", text)
        text = text.strip().strip("'\"")
        # Strip preamble phrases
        text = re.sub(
            r"^(Based on .*?research[,.]?\s*|According to .*?[,.]?\s*|"
            r"From .*?findings[,.]?\s*|The answer is[:\s]*|"
            r"My .{0,20}answer[:\s]*|After .{0,40}research[,.]?\s*|"
            r"I found that\s*|The result is[:\s]*)",
            "", text, flags=re.IGNORECASE,
        )
        # Strip "I don't know" / "unable to determine" type responses
        # (these should not be submitted as answers)
        if re.match(
            r"^(I don'?t know|Unable to determine|Cannot determine|"
            r"No .{0,30}(found|available|data)|I could not find)",
            text, re.IGNORECASE,
        ):
            # Return empty so the caller can detect this isn't a real answer
            return ""
        # Strip trailing address details from place names
        # e.g., "Renzo Gracie Jiu-Jitsu Wall Street, 22 New St, New York, NY 10005"
        # Heuristic: if there's a comma followed by a number (street address), trim
        addr_match = re.match(
            r"^(.{10,}?),\s*\d+\s+[A-Z]",
            text,
        )
        if addr_match:
            text = addr_match.group(1)
        return text.strip()

    # ── Fallback answer extraction ────────────────────────────────────

    def _fallback_answer(self, memory_path: str, question: str) -> str:
        """If the agent ran out of steps, ask the LLM to synthesise from memory."""
        memory = self._read_memory(memory_path)

        self.increment("steps")
        response = self.complete([
            Message(role="system", content=(
                "You are given a research memory document and a question. "
                "Based ONLY on the information in the memory, provide the best "
                "possible answer.\n\n"
                "RULES:\n"
                "- Output ONLY the answer — no explanation, no caveats.\n"
                "- NEVER say 'I don't know', 'Unable to determine', 'No data found', "
                "or any variant. You MUST give a concrete answer.\n"
                "- If you are uncertain, give your BEST GUESS from the data you have.\n"
                "- For place/name questions: just the name, no address or extra detail.\n"
                "- For numeric questions: just the number with appropriate units.\n"
                "- For list questions: one item per line, no bullets or numbers.\n"
            )),
            Message(role="user", content=(
                f"## Question\n{question}\n\n"
                f"## Research Memory\n{memory}"
            )),
        ], max_tokens=1024, temperature=0.0)

        answer = response.message.content.strip()
        answer = self._clean_answer(answer)
        # If _clean_answer nullified a non-answer, try once more with higher temp
        if not answer:
            response = self.complete([
                Message(role="system", content=(
                    "Answer this question with your best guess. "
                    "Output ONLY the answer value — nothing else."
                )),
                Message(role="user", content=question),
            ], max_tokens=256, temperature=0.4)
            answer = response.message.content.strip()
            answer = self._clean_answer(answer) or answer
        self._append_memory(memory_path, 999, "Fallback Answer (out of steps)", answer)
        return answer


if __name__ == "__main__":
    CompactSearchAgent().run()

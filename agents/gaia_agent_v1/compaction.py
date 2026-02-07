"""Hybrid context compaction for long-running agent sessions.

Strategy:
1. Keep the last *keep_recent* steps verbatim.
2. For older steps, truncate ``observations`` to *max_obs_chars*.
3. Archive the full observations to a JSONL file in the memory dir.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


class StepCompactor:
    """Monitors agent steps and truncates old observations to stay within
    context limits.  Safe to call every step — it no-ops when nothing
    needs compacting.
    """

    def __init__(
        self,
        memory_dir: str,
        keep_recent: int = 20,
        max_obs_chars: int = 600,
    ):
        self.memory_dir = memory_dir
        self.keep_recent = keep_recent
        self.max_obs_chars = max_obs_chars
        self._compacted: set[int] = set()     # ids of already-compacted steps

    # ------------------------------------------------------------------
    def compact(self, agent: Any) -> None:
        """Inspect the agent's step list and truncate old observations."""
        steps = self._get_steps(agent)
        if steps is None:
            return

        n = len(steps)
        if n <= self.keep_recent:
            return

        compacted = 0
        for i in range(n - self.keep_recent):
            step = steps[i]
            sid = id(step)
            if sid in self._compacted:
                continue

            obs = getattr(step, "observations", None)
            if obs and len(str(obs)) > self.max_obs_chars:
                step_num = getattr(step, "step_number", i)
                # Archive full observation
                self._archive(f"step_{step_num}_obs", str(obs)[:8000])

                # Truncate in-place
                try:
                    step.observations = (
                        str(obs)[: self.max_obs_chars]
                        + "\n[…truncated — full output archived. "
                        "Use read_notes() if needed.]"
                    )
                    compacted += 1
                except (AttributeError, TypeError):
                    pass

            self._compacted.add(sid)

        if compacted:
            print(
                f"[compaction] truncated {compacted} old observations "
                f"(steps up to {n - self.keep_recent})",
                file=sys.stderr,
                flush=True,
            )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _get_steps(agent: Any) -> list | None:
        """Retrieve step list from the agent (handles API variations)."""
        try:
            mem = getattr(agent, "memory", None)
            if mem is not None:
                steps = getattr(mem, "steps", None)
                if steps is not None:
                    return steps
        except Exception:
            pass
        try:
            logs = getattr(agent, "logs", None)
            if logs is not None:
                return logs
        except Exception:
            pass
        return None

    def _archive(self, key: str, content: str) -> None:
        path = os.path.join(self.memory_dir, "archive.jsonl")
        try:
            with open(path, "a") as f:
                json.dump({"key": key, "content": content}, f)
                f.write("\n")
        except Exception:
            pass

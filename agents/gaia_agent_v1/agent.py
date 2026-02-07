#!/usr/bin/env python3
"""GAIA Agent v1 — harness agent wrapper."""

import sys
import os

_agent_dir = os.path.dirname(os.path.abspath(__file__))
_agents_dir = os.path.dirname(_agent_dir)
if _agents_dir not in sys.path:
    sys.path.insert(0, _agents_dir)

from harness.agent import Agent


class GaiaAgentV1(Agent):
    """GAIA agent v1 — improved search, memory, compaction, better tools."""

    def run_task(self, task_id: str, task_data: dict) -> str:
        from gaia_agent_v1.runner import run_gaia_task

        self.increment("steps")

        answer = run_gaia_task(
            task_id=task_id,
            task_data=task_data,
            model_id=self.model,
            completion_callback=self.emit_completion,
        )

        self.record_tool_use("smolagents_run")
        return answer


if __name__ == "__main__":
    GaiaAgentV1().run()

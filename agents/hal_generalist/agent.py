#!/usr/bin/env python3
"""HAL Generalist Agent — smolagents-based GAIA scaffold.

Exact port of princeton-pli/hal-harness agents/hal_generalist_agent.
Uses smolagents CodeAgent with web search, web browsing, code execution,
and file reading tools.
"""

import sys
import os

# Ensure the agent's directory is on sys.path so relative imports work
# when run as a script (python agents/hal_generalist/agent.py)
_agent_dir = os.path.dirname(os.path.abspath(__file__))
_agents_dir = os.path.dirname(_agent_dir)
if _agents_dir not in sys.path:
    sys.path.insert(0, _agents_dir)

from harness.agent import Agent


class HALGeneralistAgent(Agent):
    """GAIA agent using smolagents CodeAgent with tools."""

    def run_task(self, task_id: str, task_data: dict) -> str:
        from hal_generalist.smolagents_runner import run_gaia_task

        self.increment("steps")

        answer = run_gaia_task(
            task_id=task_id,
            task_data=task_data,
            model_id=self.model,  # comes from --model flag / HARNESS_MODEL env
            completion_callback=self.emit_completion,  # usage/cost contract
        )

        self.record_tool_use("smolagents_run")
        return answer


if __name__ == "__main__":
    HALGeneralistAgent().run()

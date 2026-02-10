#!/usr/bin/env python3
"""Example agent demonstrating RunRecord metadata tracking.

This agent shows how to use the new metadata features for
ablation studies and agent configuration tracking.
"""

from harness.agent import Agent
from harness.providers.base import Message


class ExampleMetadataAgent(Agent):
    """Example agent that sets metadata for ablation analysis."""
    
    def __init__(self):
        super().__init__()
        
        # Set agent configuration metadata for tracking across runs
        # This is what you'd vary between runs for ablation studies
        self.set_metadata("planning_strategy", "react")
        self.set_metadata("tools", ["llm"])
        self.set_metadata("prompt_version", "v1.0")
        self.set_metadata("max_reasoning_steps", 3)
        self.set_metadata("temperature", 0.7)
    
    def run_task(self, task_id: str, task_data: dict) -> str:
        """Solve the task using simple prompting."""
        
        question = task_data.get("question", "")
        self.increment("steps")
        
        # Simple prompt for demonstration
        response = self.complete([
            Message(role="system", content="You are a helpful assistant. Answer questions briefly and accurately."),
            Message(role="user", content=f"Question: {question}")
        ])
        
        # Track some metrics
        self.record_tool_use("llm_call", model=self.model)
        self.metric("response_length", len(response.message.content))
        
        return response.message.content


if __name__ == "__main__":
    ExampleMetadataAgent().run()
#!/usr/bin/env python3
"""Example agent that demonstrates metrics tracking."""

from harness.agent import Agent
from harness.providers.base import Message


class MetricsExampleAgent(Agent):
    """
    An agent that tracks various metrics.
    
    This example shows how to use:
    - self.increment() for counters
    - self.metric() for arbitrary values
    - self.record_tool_use() for tool tracking
    """
    
    def run_task(self, task_id: str, task_data: dict) -> str:
        question = task_data.get("question", "")
        
        # Track that we started
        self.increment("steps")
        
        # Simulate deciding which "tools" to use
        if "calculate" in question.lower() or any(op in question for op in ["+", "-", "*", "/"]):
            self.record_tool_use("calculator")
        
        # Make an LLM call
        self.record_tool_use("llm_call")
        self.increment("steps")
        
        response = self.complete([
            Message(role="system", content="You are a helpful assistant. Answer concisely."),
            Message(role="user", content=question),
        ])
        
        answer = response.message.content.strip()
        
        # Track some metrics about the response
        self.metric("response_length", len(answer))
        self.metric("model_used", response.model or "unknown")
        
        # Final step count
        self.increment("steps")
        
        return answer


if __name__ == "__main__":
    MetricsExampleAgent().run()

#!/usr/bin/env python3
"""
Simple QA agent - makes a single LLM call to answer questions.

This is a minimal agent showing how to use the Agent helper class.
It makes one LLM call per task with no tools or multi-turn reasoning.
"""

import sys
import os

# Add parent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from harness.agent import Agent
from harness.providers.base import Message


class SimpleQAAgent(Agent):
    """Simple QA agent that answers questions with a single LLM call."""
    
    def run_task(self, task_id: str, task_data: dict) -> str:
        question = task_data.get("question", str(task_data))
        
        self.log(type="thinking", content=f"Answering: {question[:100]}...")
        
        response = self.complete([
            Message(
                role="system",
                content="Answer the question concisely and directly. Give only the answer, no explanation.",
            ),
            Message(role="user", content=question),
        ])
        
        return response.message.content


if __name__ == "__main__":
    SimpleQAAgent().run()

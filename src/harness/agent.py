"""Agent helper class for Python scaffolds."""

import sys
import json
from typing import Any

from .providers import get_provider, configure
from .providers.base import Message, CompletionRequest, CompletionResponse
from .logging import StdoutLogger


class Agent:
    """
    Base class for Python agents with built-in provider access.
    
    Subclass this and implement run_task() to create an agent.
    LLM completions are automatically logged.
    
    Example:
        class MyAgent(Agent):
            def run_task(self, task_id: str, task_data: dict) -> str:
                response = self.complete([
                    Message(role="user", content=task_data["question"])
                ])
                return response.content
        
        if __name__ == "__main__":
            MyAgent().run()
    """
    
    def __init__(self):
        self.provider = None
        self._logger = StdoutLogger()
    
    def run(self) -> None:
        """Main loop: read tasks from stdin, write results to stdout."""
        # Configure provider with stdout logging
        configure(logger=self._logger)
        self.provider = get_provider()
        
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if msg.get("method") == "run_task":
                task_id = msg["params"]["task_id"]
                task_data = msg["params"]["task_data"]
                
                try:
                    submission = self.run_task(task_id, task_data)
                    self._send_result(task_id, submission, msg.get("id"))
                except Exception as e:
                    self._send_error(str(e), msg.get("id"))
    
    def run_task(self, task_id: str, task_data: dict[str, Any]) -> str:
        """
        Override this method to implement your agent logic.
        
        Args:
            task_id: Unique identifier for the task
            task_data: Task data (usually contains "question" key)
        
        Returns:
            The submission/answer as a string
        """
        raise NotImplementedError("Subclass must implement run_task()")
    
    def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Make an LLM completion request.
        
        The request and response are automatically logged.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional arguments (max_tokens, temperature, tools, etc.)
        
        Returns:
            CompletionResponse with the model's response
        """
        request = CompletionRequest(messages=messages, **kwargs)
        return self.provider.complete(request)
    
    def log(self, event_type: str = "info", **data: Any) -> None:
        """
        Emit a custom log entry.
        
        This will be captured by the harness and written to the trace file.
        """
        self._logger.log(event_type, **data)
    
    def _send_result(self, task_id: str, submission: str, req_id: Any) -> None:
        """Send the final result to the harness."""
        msg = {
            "jsonrpc": "2.0",
            "result": {"task_id": task_id, "submission": submission},
            "id": req_id,
        }
        print(json.dumps(msg))
        sys.stdout.flush()
    
    def _send_error(self, error: str, req_id: Any) -> None:
        """Send an error to the harness."""
        msg = {
            "jsonrpc": "2.0",
            "error": {"code": -1, "message": error},
            "id": req_id,
        }
        print(json.dumps(msg))
        sys.stdout.flush()

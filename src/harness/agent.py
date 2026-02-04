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
    
    Metrics tracking is built-in. Use self.metric(), self.increment(), 
    or self.record_tool_use() to track KPIs that get aggregated in run.json.
    
    Example:
        class MyAgent(Agent):
            def run_task(self, task_id: str, task_data: dict) -> str:
                self.increment("steps")                
                response = self.complete([
                    Message(role="user", content=task_data["question"])
                ])
                self.record_tool_use("llm_call")
                return response.content
        
        if __name__ == "__main__":
            MyAgent().run()
    """
    
    def __init__(self):
        self.provider = None
        self._logger = StdoutLogger()
        self._metrics: dict[str, Any] = {}  # Per-task metrics
        self._tool_sequence: list[str] = []  # Track tool usage order
    
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
                
                # Reset metrics for this task
                self._metrics = {}
                self._tool_sequence = []
                
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
    
    # =========================================================================
    # Metrics tracking
    # =========================================================================
    
    def metric(self, name: str, value: Any) -> None:
        """
        Set a metric value for the current task.
        
        Args:
            name: Metric name (e.g., "final_answer_confidence")
            value: Any JSON-serializable value
        
        Example:
            self.metric("confidence", 0.95)
            self.metric("sources_used", ["web", "calculator"])
        """
        self._metrics[name] = value
    
    def increment(self, name: str, amount: int = 1) -> int:
        """
        Increment a counter metric.
        
        Args:
            name: Counter name (e.g., "steps", "llm_calls", "retries")
            amount: Amount to increment (default: 1)
        
        Returns:
            New counter value
            
        Example:
            self.increment("steps")
            self.increment("tokens_used", 150)
        """
        self._metrics[name] = self._metrics.get(name, 0) + amount
        return self._metrics[name]
    
    def record_tool_use(self, tool_name: str, **details: Any) -> None:
        """
        Record a tool usage, tracking both count and sequence.
        
        Args:
            tool_name: Name of tool used (e.g., "calculator", "web_search")
            **details: Optional details about the tool call
        
        Example:
            self.record_tool_use("web_search", query="weather NYC")
            self.record_tool_use("calculator", expression="2+2")
        """
        # Track sequence
        self._tool_sequence.append(tool_name)
        self._metrics["tool_sequence"] = self._tool_sequence
        
        # Increment tool counter
        tool_counts_key = "tool_counts"
        if tool_counts_key not in self._metrics:
            self._metrics[tool_counts_key] = {}
        self._metrics[tool_counts_key][tool_name] = \
            self._metrics[tool_counts_key].get(tool_name, 0) + 1
        
        # Log tool call if details provided
        if details:
            self.log("tool_call", tool=tool_name, **details)
    
    def get_metrics(self) -> dict[str, Any]:
        """Get all metrics for the current task."""
        return self._metrics.copy()
    
    # =========================================================================
    # JSON-RPC communication
    # =========================================================================
    
    def _send_result(self, task_id: str, submission: str, req_id: Any) -> None:
        """Send the final result to the harness."""
        result = {"task_id": task_id, "submission": submission}
        
        # Include metrics if any were recorded
        if self._metrics:
            result["metrics"] = self._metrics
        
        msg = {
            "jsonrpc": "2.0",
            "result": result,
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

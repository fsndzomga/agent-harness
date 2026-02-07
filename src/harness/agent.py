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
    
    **Completion logging contract:**
    
    The harness needs ``type: "completion"`` events in the trace to compute
    usage/cost.  There are two ways to satisfy this:
    
    1. Use ``self.complete()`` — logging is automatic (via the provider).
    2. Use an external framework (smolagents, langchain, …) and call
       ``self.emit_completion()`` after each LLM call.
    
    If ``run_task()`` finishes without *any* completion events being
    logged, a warning is printed so the developer notices immediately.
    
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
        self._completion_count: int = 0  # Tracks completions per task
    
    @property
    def model(self) -> str:
        """Get the model from HARNESS_MODEL env var."""
        import os
        return os.environ.get("HARNESS_MODEL", "")
    
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
                
                # Reset per-task state
                self._metrics = {}
                self._tool_sequence = []
                self._completion_count = 0
                
                try:
                    submission = self.run_task(task_id, task_data)
                    if self._completion_count == 0 and submission:
                        print(
                            "WARNING: No completion events were logged during "
                            "run_task(). Usage and cost will be zero in "
                            "run.json. To fix: call self.emit_completion() "
                            "after each LLM call, or use self.complete() "
                            "which logs automatically.",
                            file=sys.stderr, flush=True,
                        )
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
        
        The request and response are automatically logged via the provider's
        ``BaseProvider.complete()`` wrapper.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional arguments (max_tokens, temperature, tools, etc.)
        
        Returns:
            CompletionResponse with the model's response
        """
        self._completion_count += 1
        request = CompletionRequest(messages=messages, **kwargs)
        return self.provider.complete(request)
    
    def emit_completion(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
        provider: str = "external",
        raw_request: dict | None = None,
        raw_response: dict | None = None,
    ) -> None:
        """Emit a completion event for usage/cost tracking.
        
        **Call this for every LLM completion** when your agent uses an
        external framework (smolagents, langchain, crewai, …) instead of
        ``self.complete()``.
        
        The harness derives all usage and cost metrics from ``completion``
        trace events.  If none are emitted, ``run.json`` will show zeros.
        
        Args:
            model: Model identifier (e.g. ``"openrouter/deepseek/deepseek-chat-v3-0324"``)
            input_tokens: Prompt / input token count
            output_tokens: Completion / output token count
            cost_usd: Cost for this single completion (if known)
            latency_ms: Request latency in milliseconds
            provider: Provider name for attribution (e.g. ``"litellm/openrouter"``)
            raw_request: Full request dict (optional, for debugging)
            raw_response: Full response dict (optional, for debugging)
        
        Example::
        
            class MyLangchainAgent(Agent):
                def run_task(self, task_id, task_data):
                    result = my_chain.invoke(task_data["question"])
                    # Report tokens so the harness can track usage
                    self.emit_completion(
                        model="gpt-4o",
                        input_tokens=result.usage.prompt_tokens,
                        output_tokens=result.usage.completion_tokens,
                    )
                    return result.content
        """
        self._completion_count += 1
        
        response = dict(raw_response) if raw_response else {}
        if "usage" not in response:
            response["usage"] = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        if cost_usd and "_cost_usd" not in response:
            response["_cost_usd"] = cost_usd
        
        request = raw_request or {"model": model}
        
        self._logger.log_completion(
            provider=provider,
            request=request,
            response=response,
            latency_ms=latency_ms,
        )
    
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

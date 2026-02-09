"""JSON-RPC protocol definitions for agent communication."""

from dataclasses import dataclass, field
from typing import Any, Optional
import json


@dataclass
class Task:
    """A single benchmark task.

    Core fields (``id``, ``data``) are required.  The optional fields
    (``execution_mode``, ``environment``, ``metadata``) were added by the
    v2 architecture.  Existing code that creates ``Task(id=..., data=...)``
    continues to work unchanged.
    """
    id: str
    data: dict[str, Any]
    # v2 fields – optional for backward compatibility
    execution_mode: Optional[str] = None   # "direct", "interactive", etc.
    environment: Any = None                # TaskEnvironment (from benchmarks.base)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_jsonrpc(self) -> str:
        """Format as JSON-RPC request."""
        return json.dumps({
            "jsonrpc": "2.0",
            "method": "run_task",
            "params": {"task_id": self.id, "task_data": self.data},
            "id": 1,
        })
    
    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        """Create Task from dictionary."""
        return cls(
            id=d.get("id", "task_1"),
            data=d.get("data", d),
            execution_mode=d.get("execution_mode"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Submission:
    """A submission from an agent."""
    task_id: str
    answer: str
    trace: list[dict] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)  # Agent-reported KPIs
    
    @classmethod
    def from_jsonrpc(cls, line: str) -> "Submission":
        """Parse JSON-RPC response."""
        msg = json.loads(line)
        if "error" in msg:
            error = msg["error"]
            raise RuntimeError(f"Agent error: {error.get('message', error)}")
        result = msg["result"]
        return cls(
            task_id=result["task_id"],
            answer=result["submission"],
            trace=result.get("trace", []),
            metrics=result.get("metrics", {}),
        )

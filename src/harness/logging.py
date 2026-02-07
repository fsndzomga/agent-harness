"""JSONL trace logging - captures raw provider responses.

Completion logging contract
===========================

The harness computes usage/cost from ``type: "completion"`` trace events.
There are three ways to emit them:

1. **Harness providers** (``BaseProvider``) — automatic, handled by
   ``BaseProvider.complete()`` → ``TraceLogger.log_completion()``.

2. **Agent.emit_completion()** — for agents that extend ``Agent`` but
   use an external LLM framework (smolagents, langchain, …).

3. **emit_completion()** (this module) — standalone helper for agents
   that do NOT extend ``Agent`` (e.g. shell scripts, Go agents) but
   *can* import ``harness``.

For agents in other languages, emit the following JSON-RPC on stdout::

    {"jsonrpc":"2.0","method":"log","params":{"type":"completion","timestamp":…,
     "provider":"…","request":{"model":"…"},"response":{"model":"…",
     "usage":{"prompt_tokens":…,"completion_tokens":…,"total_tokens":…}},
     "latency_ms":…}}
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import TextIO, Any


@dataclass
class TraceLogger:
    """Writes agent traces to JSONL file."""
    
    output_path: Path
    run_id: str
    _file: TextIO = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False)
    
    def __post_init__(self):
        self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "a")
    
    def log(self, event_type: str, **data: Any) -> None:
        """Log a harness-generated event."""
        record = {
            "timestamp": time.time(),
            "run_id": self.run_id,
            "type": event_type,
            **data,
        }
        self._write(record)
    
    def log_raw(self, data: dict) -> None:
        """Log agent-provided data directly (raw provider responses, etc.)."""
        if "timestamp" not in data:
            data = {**data, "timestamp": time.time()}
        data["run_id"] = self.run_id
        self._write(data)
    
    def log_completion(
        self,
        provider: str,
        request: dict,
        response: dict,
        latency_ms: int,
    ) -> None:
        """Log a completion request/response (raw, no transformation)."""
        self._write({
            "timestamp": time.time(),
            "run_id": self.run_id,
            "type": "completion",
            "provider": provider,
            "request": request,
            "response": response,
            "latency_ms": latency_ms,
        })
    
    def _write(self, record: dict) -> None:
        if self._closed:
            return
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
    
    def close(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True
    
    def __enter__(self) -> "TraceLogger":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()


class StdoutLogger:
    """Logger that emits to stdout for harness capture (used by agents)."""
    
    def log(self, event_type: str, **data: Any) -> None:
        self.log_raw({"type": event_type, **data})
    
    def log_raw(self, data: dict) -> None:
        import sys
        if "timestamp" not in data:
            data = {**data, "timestamp": time.time()}
        msg = {"jsonrpc": "2.0", "method": "log", "params": data}
        print(json.dumps(msg))
        sys.stdout.flush()
    
    def log_completion(
        self,
        provider: str,
        request: dict,
        response: dict,
        latency_ms: int,
    ) -> None:
        """Log a completion - automatically captured by harness."""
        self.log_raw({
            "type": "completion",
            "provider": provider,
            "request": request,
            "response": response,
            "latency_ms": latency_ms,
        })


# =====================================================================
# Standalone helper — for agents that can't use Agent.emit_completion()
# =====================================================================

def emit_completion(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: int = 0,
    provider: str = "external",
) -> None:
    """Emit a completion event on stdout for the harness to capture.

    Use this when your agent runs as a subprocess (any language) and
    you need to report LLM usage without going through a harness
    ``Provider`` or ``Agent`` class.

    The JSON-RPC line printed to stdout matches the format that the
    harness runner expects and that ``aggregate_trace_file`` parses.

    Args:
        model: Model identifier (e.g. ``"gpt-4o"``)
        input_tokens: Prompt / input token count
        output_tokens: Completion / output token count
        cost_usd: Cost for this single completion (if known)
        latency_ms: Request latency in milliseconds
        provider: Provider name for attribution

    Example (Python agent without ``Agent`` base class)::

        from harness.logging import emit_completion

        response = my_llm_call(prompt)
        emit_completion(
            model="gpt-4o",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            cost_usd=response.cost,
        )
    """
    import sys

    response: dict = {
        "model": model,
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    if cost_usd:
        response["_cost_usd"] = cost_usd

    data = {
        "type": "completion",
        "timestamp": time.time(),
        "provider": provider,
        "request": {"model": model},
        "response": response,
        "latency_ms": latency_ms,
    }
    msg = {"jsonrpc": "2.0", "method": "log", "params": data}
    print(json.dumps(msg))
    sys.stdout.flush()

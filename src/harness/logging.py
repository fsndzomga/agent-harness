"""JSONL trace logging - captures raw provider responses."""

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

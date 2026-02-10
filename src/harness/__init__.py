"""Agent Harness - A local-first, lightweight harness for AI agent evaluations."""

__version__ = "0.1.0"

from .protocol import Task, Submission
from .runner import AgentRunner
from .logging import emit_completion
from .run_metadata import RunRecord, RunRecordStore, create_run_record

__all__ = [
    "Task", "Submission", "AgentRunner", "emit_completion",
    "RunRecord", "RunRecordStore", "create_run_record",
    "__version__",
]

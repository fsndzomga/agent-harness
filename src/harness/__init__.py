"""Agent Harness - A local-first, lightweight harness for AI agent evaluations."""

__version__ = "0.1.0"

from .protocol import Task, Submission
from .runner import AgentRunner
from .logging import emit_completion

__all__ = ["Task", "Submission", "AgentRunner", "emit_completion", "__version__"]

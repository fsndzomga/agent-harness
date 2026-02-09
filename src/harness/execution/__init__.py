"""Execution infrastructure for the v2 architecture.

Provides the ``TaskOrchestrator`` that routes tasks to the correct
execution flow based on their ``ExecutionMode``.
"""

from .orchestrator import TaskOrchestrator

__all__ = ["TaskOrchestrator"]

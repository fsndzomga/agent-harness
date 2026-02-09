"""TaskOrchestrator — mode-aware dispatch for benchmark tasks.

Routes each task to the correct execution flow based on its
``ExecutionMode``.  The orchestrator handles:

1. Environment setup (via ``benchmark.setup_environment()``)
2. Mode-specific dispatch to the agent
3. Grading (via ``benchmark.grade()``)
4. Environment teardown (via ``benchmark.teardown_environment()``)
"""

from __future__ import annotations

import logging
from typing import Any

from ..protocol import Task
from ..benchmarks.base import (
    Benchmark,
    ExecutionMode,
    ExecutionContext,
    GradeResult,
)
from ..agents.base import AgentBase

logger = logging.getLogger(__name__)


class TaskOrchestrator:
    """Routes tasks to the correct execution flow based on ExecutionMode.

    P0 implements DIRECT mode dispatch.  Other modes (CODE_SUBMIT,
    INTERACTIVE, CONVERSATIONAL, TOOL_USE, GUI_AGENT) will be added
    in subsequent milestones (P1–P5).
    """

    def run(
        self,
        task: Task,
        agent: AgentBase,
        benchmark: Benchmark,
    ) -> GradeResult:
        """Execute a single task through the full lifecycle.

        1. ``benchmark.setup_environment(task)`` → ``ExecutionContext``
        2. Dispatch to the agent based on ``benchmark.execution_mode``
        3. ``benchmark.grade(task, result, context)`` → ``GradeResult``
        4. ``benchmark.teardown_environment(context)``

        Returns:
            GradeResult from the benchmark's grading.
        """
        mode = benchmark.execution_mode
        context = benchmark.setup_environment(task)

        try:
            result = self._dispatch(mode, task, agent, context)
            grade = benchmark.grade(task, result, context)
            return grade
        finally:
            benchmark.teardown_environment(context)

    def run_all(
        self,
        benchmark: Benchmark,
        agent: AgentBase,
        *,
        task_ids: list[str] | None = None,
    ) -> list[GradeResult]:
        """Run all (or selected) tasks for a benchmark.

        Args:
            benchmark: The benchmark to evaluate.
            agent: The agent to evaluate.
            task_ids: Optional list of task IDs to run (subset).
                      If None, runs all tasks.

        Returns:
            List of GradeResult for each task.
        """
        tasks = benchmark.get_tasks()
        if task_ids is not None:
            id_set = set(task_ids)
            tasks = [t for t in tasks if t.id in id_set]

        results: list[GradeResult] = []
        for task in tasks:
            try:
                grade = self.run(task, agent, benchmark)
                results.append(grade)
            except Exception as exc:
                logger.error("Task %s failed: %s", task.id, exc)
                results.append(GradeResult(
                    task_id=task.id,
                    passed=False,
                    score=0.0,
                    expected="",
                    actual=str(exc),
                    method="error",
                    details={"error": str(exc)},
                ))
        return results

    # ------------------------------------------------------------------
    # Mode dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        mode: ExecutionMode,
        task: Task,
        agent: AgentBase,
        context: ExecutionContext,
    ) -> Any:
        """Route to the correct agent method based on execution mode."""
        match mode:
            case ExecutionMode.DIRECT:
                return self._run_direct(task, agent, context)
            case ExecutionMode.CODE_SUBMIT:
                return self._run_code_submit(task, agent, context)
            case ExecutionMode.INTERACTIVE:
                return self._run_interactive(task, agent, context)
            case ExecutionMode.CONVERSATIONAL:
                return self._run_conversational(task, agent, context)
            case ExecutionMode.TOOL_USE:
                return self._run_tool_use(task, agent, context)
            case ExecutionMode.GUI_AGENT:
                return self._run_gui_agent(task, agent, context)
            case _:
                raise ValueError(f"Unknown execution mode: {mode}")

    # ── A: DIRECT ────────────────────────────────────────────────

    def _run_direct(
        self,
        task: Task,
        agent: AgentBase,
        context: ExecutionContext,
    ) -> str:
        """Question → Answer.  The simplest archetype."""
        return agent.run_task(task.id, task.data)

    # ── B: CODE_SUBMIT ───────────────────────────────────────────

    def _run_code_submit(
        self,
        task: Task,
        agent: AgentBase,
        context: ExecutionContext,
    ) -> str:
        """Problem → Code → Container Tests.  Implemented in P1."""
        raise NotImplementedError(
            "CODE_SUBMIT mode is not yet implemented (planned for P1)"
        )

    # ── C: INTERACTIVE ───────────────────────────────────────────

    def _run_interactive(
        self,
        task: Task,
        agent: AgentBase,
        context: ExecutionContext,
    ) -> None:
        """Agent ↔ Container Shell Loop.  Implemented in P2."""
        raise NotImplementedError(
            "INTERACTIVE mode is not yet implemented (planned for P2)"
        )

    # ── D: CONVERSATIONAL ────────────────────────────────────────

    def _run_conversational(
        self,
        task: Task,
        agent: AgentBase,
        context: ExecutionContext,
    ) -> None:
        """Agent ↔ Simulated User.  Implemented in P3."""
        raise NotImplementedError(
            "CONVERSATIONAL mode is not yet implemented (planned for P3)"
        )

    # ── E: TOOL_USE ──────────────────────────────────────────────

    def _run_tool_use(
        self,
        task: Task,
        agent: AgentBase,
        context: ExecutionContext,
    ) -> str:
        """Agent + Live Tool Servers.  Implemented in P4."""
        raise NotImplementedError(
            "TOOL_USE mode is not yet implemented (planned for P4)"
        )

    # ── F: GUI_AGENT ─────────────────────────────────────────────

    def _run_gui_agent(
        self,
        task: Task,
        agent: AgentBase,
        context: ExecutionContext,
    ) -> None:
        """Agent in VM/Game World.  Implemented in P5."""
        raise NotImplementedError(
            "GUI_AGENT mode is not yet implemented (planned for P5)"
        )

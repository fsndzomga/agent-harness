"""
AssistantBench — web search and information synthesis tasks.

AssistantBench evaluates agents on open-ended web tasks requiring real-time
information retrieval and synthesis.

This is a DIRECT-mode benchmark: the agent receives a question and returns
an answer string.
"""

from __future__ import annotations

from typing import Any

from ..protocol import Task
from .base import Benchmark, ExecutionMode, ExecutionContext, GradeResult
from .graders import exact_match, normalized_match, numeric_match, fuzzy_match


class AssistantBenchBenchmark(Benchmark):
    """AssistantBench: web search and information synthesis tasks.

    Args:
        split: Dataset split — ``"validation"`` or ``"test"``.
        cache_dir: Local cache directory for downloaded data.
    """

    name = "assistant-bench"
    description = "AssistantBench — web search and information synthesis tasks"
    execution_mode = ExecutionMode.DIRECT

    def __init__(
        self,
        split: str = "validation",
        cache_dir: str | None = None,
    ):
        self.split = split
        self.cache_dir = cache_dir
        self._tasks: list[Task] | None = None
        self._answers: dict[str, str] = {}

    def get_tasks(self) -> list[Task]:
        if self._tasks is not None:
            return self._tasks

        self._load_dataset()
        return self._tasks  # type: ignore[return-value]

    def grade(self, task: Task, result: Any, context: ExecutionContext) -> GradeResult:
        task_id = task.id
        submission = str(result) if result is not None else ""
        expected = self._answers.get(task_id, "")

        if not expected:
            return GradeResult(
                task_id=task_id,
                passed=False,
                score=0.0,
                expected="[hidden]",
                actual=submission,
                method="no_answer",
            )

        # Try grading pipeline: exact → normalized → fuzzy
        sub = submission.strip()
        exp = expected.strip()

        if exact_match(sub, exp):
            return self._result(task_id, True, expected, submission, "exact")
        if normalized_match(sub, exp):
            return self._result(task_id, True, expected, submission, "normalized")
        if numeric_match(sub, exp):
            return self._result(task_id, True, expected, submission, "numeric")
        if fuzzy_match(sub, exp, threshold=0.85):
            return self._result(task_id, True, expected, submission, "fuzzy")

        return self._result(task_id, False, expected, submission, "none")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_dataset(self) -> None:
        """Load AssistantBench dataset.

        TODO: Integrate with the actual AssistantBench HuggingFace dataset
        once the dataset ID and schema are finalized.
        """
        self._tasks = []  # Placeholder — actual loading in future PR

    def _result(
        self,
        task_id: str,
        passed: bool,
        expected: str,
        actual: str,
        method: str,
    ) -> GradeResult:
        return GradeResult(
            task_id=task_id,
            passed=passed,
            score=1.0 if passed else 0.0,
            expected=expected,
            actual=actual,
            method=method,
        )

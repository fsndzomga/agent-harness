"""
BrowseComp — challenging web-search questions requiring persistent browsing.

BrowseComp consists of 1,266 questions crafted backwards from known answers,
requiring iterative search reformulation and cross-source synthesis.  Even
GPT-4o with browsing achieves ~2%; Deep Research systems reach ~52%.

Source: OpenAI, 2025
Dataset: https://github.com/openai/browsecomp

This is a DIRECT-mode benchmark.
"""

from __future__ import annotations

from typing import Any

from ..protocol import Task
from .base import Benchmark, ExecutionMode, ExecutionContext, GradeResult
from .graders import exact_match, normalized_match, numeric_match


class BrowseCompBenchmark(Benchmark):
    """BrowseComp: hard web-search Q&A.

    Args:
        split: Dataset split — ``"test"`` (public set).
        cache_dir: Local cache directory for downloaded data.
    """

    name = "browsecomp"
    description = "BrowseComp — 1,266 challenging web-search questions"
    execution_mode = ExecutionMode.DIRECT

    def __init__(
        self,
        split: str = "test",
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
        """Grade using semantic equivalence check.

        BrowseComp uses automated semantic equivalence checking against
        reference answers.  This implementation provides deterministic
        matchers; LLM-as-judge can be layered on top.
        """
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

        sub = submission.strip()
        exp = expected.strip()

        if exact_match(sub, exp):
            return self._result(task_id, True, expected, submission, "exact")
        if normalized_match(sub, exp):
            return self._result(task_id, True, expected, submission, "normalized")
        if numeric_match(sub, exp):
            return self._result(task_id, True, expected, submission, "numeric")

        return self._result(task_id, False, expected, submission, "none")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_dataset(self) -> None:
        """Load BrowseComp dataset.

        TODO: Integrate with OpenAI/browsecomp dataset.
        """
        self._tasks = []

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

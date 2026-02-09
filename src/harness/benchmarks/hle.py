"""
Humanity's Last Exam (HLE) — expert-level questions across 100+ subjects.

HLE is a benchmark of 2,500 expert-authored questions across mathematics,
sciences, humanities, and more.  Questions may be multiple-choice or
short-answer, and some include images (multimodal).

Paper: https://arxiv.org/abs/2501.14249
Dataset: https://huggingface.co/datasets/centerforaisafety/hle

This is a DIRECT-mode benchmark.
"""

from __future__ import annotations

from typing import Any

from ..protocol import Task
from .base import Benchmark, ExecutionMode, ExecutionContext, GradeResult
from .graders import exact_match, normalized_match, numeric_match


class HLEBenchmark(Benchmark):
    """Humanity's Last Exam: expert-level questions.

    Args:
        split: Dataset split — ``"test"`` (public set).
        cache_dir: Local cache directory for downloaded data.
    """

    name = "hle"
    description = "Humanity's Last Exam — 2,500 expert-level questions across 100+ subjects"
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
        """Grade using exact/normalized match.

        HLE uses automated grading with o3-mini as extractor/judge for
        the official leaderboard.  This implementation provides deterministic
        grading; LLM-as-judge can be layered on top via the grading pipeline.
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
        """Load HLE dataset.

        TODO: Integrate with centerforaisafety/hle on HuggingFace.
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

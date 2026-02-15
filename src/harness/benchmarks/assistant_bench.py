"""
AssistantBench — web search and information synthesis tasks.

AssistantBench evaluates agents on open-ended web tasks requiring real-time
information retrieval and synthesis.  Tasks span difficulty levels (Medium,
Hard) and answer types (string, numeric, list, structured JSON).

Dataset: https://huggingface.co/datasets/AssistantBench/AssistantBench
Paper:   https://arxiv.org/abs/2407.15711

This is a DIRECT-mode benchmark: the agent receives a question and returns
an answer string.
"""

from __future__ import annotations

from typing import Any

from ..protocol import Task
from .base import Benchmark, ExecutionMode, ExecutionContext, GradeResult


# ── Main benchmark class ─────────────────────────────────────────────


class AssistantBenchBenchmark(Benchmark):
    """AssistantBench: web search and information synthesis tasks.

    Loads the dataset from ``AssistantBench/AssistantBench`` on HuggingFace.

    Grading follows the official evaluation methodology:

    * **Accuracy** — 1 if the answer matches (exact / normalized / numeric), 0
      otherwise.
    * **Answer-F1** — token-level F1 for strings, set-level F1 for lists, and
      numeric closeness for numbers.
    * ``score`` is the Answer-F1 value.  ``passed`` is ``True`` when the F1
      is >= the configured threshold (default 0.8).

    Args:
        split: Dataset split — ``"validation"`` (with answers) or ``"test"``
            (answers hidden).
        difficulty: Optional difficulty filter — ``"Hard"`` or ``"Medium"``.
        cache_dir: Local cache directory for downloaded data.
        pass_threshold: Minimum F1 score to count as *passed* (default 0.8).
    """

    name = "assistant-bench"
    description = "AssistantBench — web search and information synthesis tasks"
    execution_mode = ExecutionMode.DIRECT

    HF_DATASET_ID = "AssistantBench/AssistantBench"

    def __init__(
        self,
        split: str = "validation",
        difficulty: str | None = None,
        cache_dir: str | None = None,
        pass_threshold: float = 0.8,
    ):
        self.split = split
        self.difficulty = difficulty
        self.cache_dir = cache_dir
        self.pass_threshold = pass_threshold
        self._tasks: list[Task] | None = None
        self._answers: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
                details={"reason": "No gold answer (test set or missing)"},
            )

        # ── Official AssistantBench scorer ───────────────────────
        from ._assistantbench_eval import question_scorer

        accuracy, has_ans = question_scorer(submission.strip(), expected.strip())
        score = round(float(accuracy), 4)
        passed = score >= self.pass_threshold

        return GradeResult(
            task_id=task_id,
            passed=passed,
            score=score,
            expected=expected,
            actual=submission,
            method="official_question_scorer",
            details={
                "accuracy": score,
                "has_ans": float(has_ans),
                **(self._metadata.get(task_id, {})),
            },
        )

    def get_num_tasks(self) -> int:
        """Return number of tasks without loading all task objects."""
        return len(self.get_tasks())

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self) -> None:
        """Load AssistantBench from ``AssistantBench/AssistantBench`` on HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The `datasets` package is required for AssistantBench. "
                "Install it with: pip install datasets"
            )

        ds = load_dataset(
            self.HF_DATASET_ID,
            split=self.split,
            cache_dir=self.cache_dir,
            trust_remote_code=False,
        )

        self._tasks = []
        for item in ds:
            # Difficulty filter
            difficulty = item.get("difficulty")
            if self.difficulty is not None and difficulty != self.difficulty:
                continue

            task_id = item["id"]
            question = item["task"]
            answer = item.get("answer")  # None for test split

            task_data: dict[str, Any] = {
                "question": question,
            }
            if difficulty:
                task_data["difficulty"] = difficulty
            if item.get("gold_url"):
                task_data["gold_url"] = item["gold_url"]

            self._tasks.append(
                Task(
                    id=task_id,
                    data=task_data,
                    execution_mode="direct",
                    metadata={
                        "difficulty": difficulty,
                        "set": item.get("set"),
                    },
                )
            )

            # Store answer for grading (validation only; test has None)
            if answer is not None:
                self._answers[task_id] = str(answer).strip()

            # Store extra metadata for grade details
            self._metadata[task_id] = {
                "difficulty": difficulty,
            }


class AssistantBenchHardBenchmark(AssistantBenchBenchmark):
    """AssistantBench — Hard difficulty only."""

    name = "assistant-bench-hard"
    description = "AssistantBench — Hard difficulty tasks"

    def __init__(self, **kwargs: Any):
        super().__init__(difficulty="Hard", **kwargs)


class AssistantBenchMediumBenchmark(AssistantBenchBenchmark):
    """AssistantBench — Medium difficulty only."""

    name = "assistant-bench-medium"
    description = "AssistantBench — Medium difficulty tasks"

    def __init__(self, **kwargs: Any):
        super().__init__(difficulty="Medium", **kwargs)

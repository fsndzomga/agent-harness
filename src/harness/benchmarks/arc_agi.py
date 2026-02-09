"""
ARC-AGI 1 & 2 — grid-based visual reasoning puzzles.

ARC-AGI evaluates fluid intelligence through abstract reasoning tasks
presented as input-output grid pairs.  The agent must learn the
transformation from examples and apply it to a test input.

- ARC-AGI-1: 800+ tasks (largely saturated by frontier models)
- ARC-AGI-2: 400+ tasks (much harder, best model ~31%)

Grading: exact grid match (all cells must match for all test inputs).
Binary pass/fail per task, with 3 attempts allowed per test input.

Source: https://arcprize.org/
Dataset: https://github.com/fchollet/ARC-AGI

This is a DIRECT-mode benchmark.
"""

from __future__ import annotations

import json
from typing import Any

from ..protocol import Task
from .base import Benchmark, ExecutionMode, ExecutionContext, GradeResult


class ARCAGIBenchmark(Benchmark):
    """ARC-AGI benchmark (v1 and v2).

    Args:
        version: ``1`` or ``2``.
        split: ``"evaluation"`` or ``"training"``.
        max_attempts: Number of attempts allowed per test input (default 3).
        data_dir: Path to local ARC-AGI JSON task files.
    """

    name = "arc-agi"
    description = "ARC-AGI — grid-based visual reasoning puzzles"
    execution_mode = ExecutionMode.DIRECT

    def __init__(
        self,
        version: int = 1,
        split: str = "evaluation",
        max_attempts: int = 3,
        data_dir: str | None = None,
    ):
        self.version = version
        self.split = split
        self.max_attempts = max_attempts
        self.data_dir = data_dir
        self._tasks: list[Task] | None = None
        self._expected_outputs: dict[str, list[list[list[int]]]] = {}

    def get_tasks(self) -> list[Task]:
        if self._tasks is not None:
            return self._tasks

        self._load_dataset()
        return self._tasks  # type: ignore[return-value]

    def grade(self, task: Task, result: Any, context: ExecutionContext) -> GradeResult:
        """Grade by exact grid match.

        The result should be a list of grids (one per test input), where
        each grid is a list of lists of ints.  Alternatively, the result
        can be a JSON string encoding that structure.

        With ``max_attempts > 1``, the result can be a list of lists of
        grids (one list of attempts per test input).
        """
        task_id = task.id
        expected_grids = self._expected_outputs.get(task_id, [])

        if not expected_grids:
            return GradeResult(
                task_id=task_id,
                passed=False,
                score=0.0,
                expected="[no expected output]",
                actual=str(result),
                method="no_answer",
            )

        # Parse result
        predicted = self._parse_result(result)
        if predicted is None:
            return GradeResult(
                task_id=task_id,
                passed=False,
                score=0.0,
                expected=json.dumps(expected_grids),
                actual=str(result),
                method="parse_error",
                details={"error": "Could not parse result as grid(s)"},
            )

        # Check each test input
        all_match = True
        for i, expected_grid in enumerate(expected_grids):
            matched = False
            # Get attempts for this test input
            if i < len(predicted):
                attempts = predicted[i]
                # Determine if this is a single grid or a list of attempt grids.
                # A grid is list[list[int]] — grid[0][0] is an int.
                # A list of attempts is list[grid] — attempts[0][0] is a list.
                if self._is_grid(attempts):
                    # Single attempt grid, wrap in a list
                    attempts = [attempts]

                for attempt in attempts[: self.max_attempts]:
                    if self._grids_match(attempt, expected_grid):
                        matched = True
                        break

            if not matched:
                all_match = False

        return GradeResult(
            task_id=task_id,
            passed=all_match,
            score=1.0 if all_match else 0.0,
            expected=json.dumps(expected_grids),
            actual=json.dumps(predicted) if predicted else str(result),
            method="grid_match" if all_match else "none",
        )

    # ------------------------------------------------------------------
    # Grid matching
    # ------------------------------------------------------------------

    @staticmethod
    def _is_grid(obj: Any) -> bool:
        """Check if obj is a grid: list[list[int]]."""
        if not isinstance(obj, list) or not obj:
            return False
        # A grid's first element is a row (list of ints)
        first = obj[0]
        if not isinstance(first, list) or not first:
            return False
        # If the first element of the first row is an int, it's a grid
        return isinstance(first[0], (int, float))

    @staticmethod
    def _grids_match(predicted: Any, expected: list[list[int]]) -> bool:
        """Check if two grids are identical (all cells match)."""
        if not isinstance(predicted, list):
            return False
        if len(predicted) != len(expected):
            return False
        for pred_row, exp_row in zip(predicted, expected):
            if not isinstance(pred_row, list):
                return False
            if len(pred_row) != len(exp_row):
                return False
            for p, e in zip(pred_row, exp_row):
                if p != e:
                    return False
        return True

    @staticmethod
    def _parse_result(result: Any) -> list | None:
        """Parse agent result into a list of grids or list of attempt-lists."""
        if isinstance(result, list):
            return result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self) -> None:
        """Load ARC-AGI tasks from JSON files.

        TODO: Integrate with ARC-AGI GitHub release or HuggingFace dataset.
        """
        self._tasks = []


class ARCAGI1Benchmark(ARCAGIBenchmark):
    """ARC-AGI version 1."""
    name = "arc-agi-1"
    description = "ARC-AGI-1 — 800+ grid-based visual reasoning puzzles"

    def __init__(self, **kwargs: Any):
        super().__init__(version=1, **kwargs)


class ARCAGI2Benchmark(ARCAGIBenchmark):
    """ARC-AGI version 2 (harder, best model ~31%)."""
    name = "arc-agi-2"
    description = "ARC-AGI-2 — harder grid-based visual reasoning puzzles"

    def __init__(self, **kwargs: Any):
        super().__init__(version=2, **kwargs)

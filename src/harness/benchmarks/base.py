"""Base benchmark class and grade result."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..protocol import Task


@dataclass
class GradeResult:
    """Result of grading a single submission."""
    task_id: str
    passed: bool
    score: float  # 0.0 - 1.0
    expected: str
    actual: str
    method: str  # Which grader matched (exact, normalized, numeric, fuzzy, etc.)
    details: dict[str, Any] | None = None


class Benchmark(ABC):
    """Base class for benchmarks."""
    
    name: str = "unknown"
    description: str = ""
    
    @abstractmethod
    def get_tasks(self) -> list[Task]:
        """Load and return all tasks."""
        pass
    
    @abstractmethod
    def grade(self, task_id: str, submission: str) -> GradeResult:
        """Grade a single submission."""
        pass
    
    def grade_all(self, results: list[dict]) -> list[GradeResult]:
        """
        Grade all results from a run.
        
        Args:
            results: List of dicts with "task_id" and "submission" keys
        """
        grades = []
        for r in results:
            if r.get("submission") is not None:
                grade = self.grade(r["task_id"], r["submission"])
                grades.append(grade)
        return grades
    
    def get_num_tasks(self) -> int:
        """Return number of tasks (may be expensive, override if needed)."""
        return len(self.get_tasks())

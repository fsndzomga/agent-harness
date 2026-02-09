"""GradingPipeline — unified grading with benchmark-specific delegation.

The ``GradingPipeline`` provides a layered set of shared matchers that
benchmarks can use.  The primary entry point is
``benchmark.grade(task, result, context)`` — the pipeline is a *utility*
that benchmarks call internally, not a replacement for benchmark-owned
grading.

Usage::

    pipeline = GradingPipeline()

    # Quick standalone grading (no benchmark context)
    grade = pipeline.grade_answer("Paris", "paris", task_id="q1")

    # With custom matchers
    pipeline = GradingPipeline(matchers=["exact", "normalized", "numeric"])
    grade = pipeline.grade_answer("42.0", "42", task_id="q2")
"""

from __future__ import annotations

from typing import Any, Callable

from ..benchmarks.base import GradeResult
from ..benchmarks.graders import (
    exact_match,
    normalized_match,
    numeric_match,
    fuzzy_match,
    contains_match,
)


# Matcher registry: name → callable(submission, expected) → bool
_MATCHERS: dict[str, Callable[[str, str], bool]] = {
    "exact": exact_match,
    "normalized": normalized_match,
    "numeric": numeric_match,
    "fuzzy": fuzzy_match,
    "contains": contains_match,
}

# Default matcher chain used by DIRECT benchmarks
DEFAULT_MATCHER_CHAIN = ["exact", "normalized", "numeric", "fuzzy"]


class GradingPipeline:
    """Unified grading pipeline with layered matchers.

    Benchmarks can use this directly for string-based grading or
    compose it with their own logic (container tests, claims rubrics, etc.).
    """

    def __init__(
        self,
        matchers: list[str] | None = None,
    ):
        """
        Args:
            matchers: Ordered list of matcher names to try.
                      Defaults to ``["exact", "normalized", "numeric", "fuzzy"]``.
        """
        names = matchers or DEFAULT_MATCHER_CHAIN
        self._matchers: list[tuple[str, Callable[[str, str], bool]]] = []
        for name in names:
            fn = _MATCHERS.get(name)
            if fn is None:
                raise ValueError(
                    f"Unknown matcher: {name!r}. "
                    f"Available: {sorted(_MATCHERS.keys())}"
                )
            self._matchers.append((name, fn))

    @property
    def matcher_names(self) -> list[str]:
        """Names of the matchers in this pipeline, in order."""
        return [name for name, _ in self._matchers]

    def match(self, submission: str, expected: str) -> tuple[bool, str]:
        """Try matchers in order, return ``(passed, method)`` on first hit.

        Returns ``(False, "none")`` if no matcher matches.
        """
        for name, fn in self._matchers:
            try:
                if fn(submission, expected):
                    return True, name
            except Exception:
                continue
        return False, "none"

    def grade_answer(
        self,
        submission: str,
        expected: str,
        *,
        task_id: str = "",
    ) -> GradeResult:
        """Grade a string answer using the layered matcher pipeline.

        This is a convenience method for DIRECT-mode benchmarks that
        compare string answers.
        """
        passed, method = self.match(submission, expected)
        return GradeResult(
            task_id=task_id,
            passed=passed,
            score=1.0 if passed else 0.0,
            expected=expected,
            actual=submission,
            method=method,
        )

    # ------------------------------------------------------------------
    # Extensibility — register custom matchers
    # ------------------------------------------------------------------

    @staticmethod
    def register_matcher(name: str, fn: Callable[[str, str], bool]) -> None:
        """Register a custom matcher globally.

        Registered matchers can be referenced by name when constructing
        a ``GradingPipeline``.
        """
        _MATCHERS[name] = fn

    @staticmethod
    def available_matchers() -> list[str]:
        """Return sorted list of all registered matcher names."""
        return sorted(_MATCHERS.keys())

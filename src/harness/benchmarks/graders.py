"""Grading utilities — Grader ABC, built-in graders, and registry.

Architecture
============
Every grader is a subclass of ``Grader`` (an ABC).  Graders are registered by
name in ``GRADER_REGISTRY`` so they can be looked up from the CLI.

A single CLI ``--grader`` flag accepts **comma-separated** names.  Each name
produces its **own** score section in ``run.json`` and ``grades.json``.

Legacy helpers (``grade_with_pipeline``, ``get_grader_preset``, etc.) are kept
for backward compatibility but new code should prefer the class-based API.
"""

from __future__ import annotations

import re
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable

from .base import GradeResult


# ============================================================================
# Grader ABC
# ============================================================================

class Grader(ABC):
    """Base class for all graders.

    Subclasses MUST implement ``name`` (property) and ``grade_one``.
    The ``grade`` convenience method builds a full ``GradeResult``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier used as the key in ``scores`` / ``grades.json``."""
        ...

    @abstractmethod
    def grade_one(
        self,
        submission: str,
        expected: str,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        """Grade a single submission.

        Returns:
            (passed, method_detail)  — *method_detail* describes which
            sub-check matched (e.g. ``"exact"``, ``"llm_judge"``).
        """
        ...

    # -- convenience -----------------------------------------------------------

    def grade(
        self,
        task_id: str,
        submission: str,
        expected: str,
        **kwargs: Any,
    ) -> GradeResult:
        """Build a full ``GradeResult`` for one task."""
        passed, method = self.grade_one(submission, expected, **kwargs)
        return GradeResult(
            task_id=task_id,
            passed=passed,
            score=1.0 if passed else 0.0,
            expected=expected,
            actual=submission,
            method=method,
        )

    def grade_batch(
        self,
        submissions: list[dict[str, Any]],
    ) -> list[GradeResult]:
        """Grade many submissions.

        Each dict must have ``task_id``, ``submission``, ``expected``.
        Extra keys are forwarded as ``**kwargs`` to ``grade_one``.
        """
        results: list[GradeResult] = []
        for sub in submissions:
            extra = {
                k: v
                for k, v in sub.items()
                if k not in ("task_id", "submission", "expected")
            }
            results.append(
                self.grade(
                    task_id=sub["task_id"],
                    submission=sub["submission"],
                    expected=sub["expected"],
                    **extra,
                )
            )
        return results

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


# ============================================================================
# Match functions (kept for backward compat and used by Pipeline/preset graders)
# ============================================================================

def exact_match(submission: str, expected: str) -> bool:
    """Exact string match after strip."""
    return submission.strip() == expected.strip()


def normalized_match(submission: str, expected: str) -> bool:
    """Match after normalizing whitespace and case."""
    def normalize(s: str) -> str:
        return re.sub(r'\s+', ' ', s.lower().strip())
    return normalize(submission) == normalize(expected)


def numeric_match(
    submission: str,
    expected: str,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> bool:
    """Match numeric values with tolerance.

    Extracts numbers from both strings and compares the last number found.
    """
    try:
        sub_nums = re.findall(r'-?\d+\.?\d*', submission)
        exp_nums = re.findall(r'-?\d+\.?\d*', expected)
        if not sub_nums or not exp_nums:
            return False
        sub_val = float(sub_nums[-1])
        exp_val = float(exp_nums[-1])
        return math.isclose(sub_val, exp_val, rel_tol=rtol, abs_tol=atol)
    except (ValueError, IndexError):
        return False


def fuzzy_match(submission: str, expected: str, threshold: float = 0.9) -> bool:
    """Fuzzy string match using sequence similarity."""
    sub_norm = re.sub(r'\s+', ' ', submission.lower().strip())
    exp_norm = re.sub(r'\s+', ' ', expected.lower().strip())
    return SequenceMatcher(None, sub_norm, exp_norm).ratio() >= threshold


def contains_match(submission: str, expected: str) -> bool:
    """Check if submission contains the expected answer."""
    return expected.lower().strip() in submission.lower().strip()


# ============================================================================
# Concrete Graders
# ============================================================================

class ExactGrader(Grader):
    """Exact string match after stripping whitespace."""

    @property
    def name(self) -> str:
        return "exact"

    def grade_one(self, submission: str, expected: str, **kw: Any) -> tuple[bool, str]:
        return exact_match(submission, expected), "exact"


class NormalizedGrader(Grader):
    """Case- and whitespace-normalized match."""

    @property
    def name(self) -> str:
        return "normalized"

    def grade_one(self, submission: str, expected: str, **kw: Any) -> tuple[bool, str]:
        return normalized_match(submission, expected), "normalized"


class NumericGrader(Grader):
    """Numeric match with configurable tolerance."""

    def __init__(self, rtol: float = 1e-3, atol: float = 1e-6):
        self.rtol = rtol
        self.atol = atol

    @property
    def name(self) -> str:
        return "numeric"

    def grade_one(self, submission: str, expected: str, **kw: Any) -> tuple[bool, str]:
        return numeric_match(submission, expected, rtol=self.rtol, atol=self.atol), "numeric"


class ContainsGrader(Grader):
    """Checks whether the expected answer is contained in the submission."""

    @property
    def name(self) -> str:
        return "contains"

    def grade_one(self, submission: str, expected: str, **kw: Any) -> tuple[bool, str]:
        return contains_match(submission, expected), "contains"


class FuzzyGrader(Grader):
    """Fuzzy sequence-matcher grader."""

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "fuzzy"

    def grade_one(self, submission: str, expected: str, **kw: Any) -> tuple[bool, str]:
        return fuzzy_match(submission, expected, threshold=self.threshold), "fuzzy"


class StrictGrader(Grader):
    """Exact then normalized — no lenient matchers."""

    @property
    def name(self) -> str:
        return "strict"

    def grade_one(self, submission: str, expected: str, **kw: Any) -> tuple[bool, str]:
        if exact_match(submission, expected):
            return True, "exact"
        if normalized_match(submission, expected):
            return True, "normalized"
        return False, "none"


# ============================================================================
# Pipeline Grader (default)
# ============================================================================

DEFAULT_MATCHERS: list[tuple[str, Callable[[str, str], bool]]] = [
    ("exact", exact_match),
    ("normalized", normalized_match),
    ("numeric", numeric_match),
]

ALL_MATCHERS: list[tuple[str, Callable[[str, str], bool]]] = [
    ("exact", exact_match),
    ("normalized", normalized_match),
    ("numeric", numeric_match),
    ("contains", contains_match),
    ("fuzzy", fuzzy_match),
]


class PipelineGrader(Grader):
    """Tries a sequence of matchers, returns on the first pass.

    With the default matchers this is equivalent to the legacy
    ``grade_with_pipeline()`` behaviour.
    """

    def __init__(
        self,
        matchers: list[tuple[str, Callable[[str, str], bool]]] | None = None,
        *,
        grader_name: str = "default",
    ):
        self._matchers = matchers or DEFAULT_MATCHERS
        self._name = grader_name

    @property
    def name(self) -> str:
        return self._name

    def grade_one(self, submission: str, expected: str, **kw: Any) -> tuple[bool, str]:
        for matcher_name, matcher_fn in self._matchers:
            try:
                if matcher_fn(submission, expected):
                    return True, matcher_name
            except Exception:
                continue
        return False, "none"


# ============================================================================
# LLM-as-Judge Grader
# ============================================================================

LLM_JUDGE_PROMPT = """You are a strict evaluator. Determine whether a submission matches the expected answer.

Expected answer: {expected}
Submission: {submission}

Rules — apply these strictly:
1. The submission must convey the SAME factual content as the expected answer.
2. If both are numbers, they must be equal (or within trivial rounding, e.g. 3.14 vs 3.141).
   A different number is INCORRECT — do NOT assume unit conversions or scale differences.
3. Minor formatting differences are acceptable (capitalization, punctuation, whitespace).
4. If the submission contains extra text but includes the correct answer, that is CORRECT.
5. If the submission is a completely different word, value, or concept, it is INCORRECT.
6. When in doubt, answer INCORRECT.

Reply with ONLY "CORRECT" or "INCORRECT" (no explanation)."""


class LLMJudge:
    """Low-level LLM-as-judge helper (used by the Grader subclasses)."""

    def __init__(self, model: str | None = None):
        self.model = model
        self._provider = None

    @property
    def provider(self):
        if self._provider is None:
            from ..providers import get_provider
            self._provider = get_provider(model=self.model)
        return self._provider

    def grade(
        self,
        submission: str,
        expected: str,
        question: str = "",
    ) -> tuple[bool, str, str]:
        """Returns (passed, method, raw_answer)."""
        from ..providers.base import Message, CompletionRequest

        prompt = LLM_JUDGE_PROMPT.format(
            expected=expected,
            submission=submission,
        )
        if question:
            prompt = f"Question: {question}\n\n{prompt}"

        try:
            response = self.provider.complete(
                CompletionRequest(
                    messages=[Message(role="user", content=prompt)],
                    max_tokens=10,
                    temperature=0.0,
                )
            )
            answer = response.message.content.strip().upper()
            passed = answer.startswith("CORRECT") or answer == "CORRECT"
            return passed, "llm_judge", answer
        except Exception as e:
            return False, "llm_judge_error", str(e)


class LLMJudgeGrader(Grader):
    """Pure LLM-as-judge grader."""

    def __init__(self, model: str | None = None):
        self._judge = LLMJudge(model=model)

    @property
    def name(self) -> str:
        return "llm"

    def grade_one(self, submission: str, expected: str, **kw: Any) -> tuple[bool, str]:
        question = kw.get("question", "")
        passed, method, _ = self._judge.grade(submission, expected, question)
        return passed, method


class LLMFallbackGrader(Grader):
    """Deterministic pipeline first, then LLM-as-judge fallback."""

    def __init__(
        self,
        model: str | None = None,
        matchers: list[tuple[str, Callable[[str, str], bool]]] | None = None,
    ):
        self._pipeline = PipelineGrader(matchers)
        self._judge = LLMJudge(model=model)

    @property
    def name(self) -> str:
        return "llm-fallback"

    def grade_one(self, submission: str, expected: str, **kw: Any) -> tuple[bool, str]:
        passed, method = self._pipeline.grade_one(submission, expected)
        if passed:
            return True, method
        question = kw.get("question", "")
        passed, method, _ = self._judge.grade(submission, expected, question)
        return passed, method


# ============================================================================
# Grader Registry
# ============================================================================

GRADER_REGISTRY: dict[str, Callable[..., Grader]] = {}
"""Map grader name → factory callable.  Factory receives **kwargs (e.g. model)."""


def register_grader(
    name: str,
    factory: Callable[..., Grader],
) -> None:
    """Register a grader factory by name."""
    GRADER_REGISTRY[name] = factory


def _register_builtins() -> None:
    """Register all built-in graders."""
    register_grader("exact", lambda **kw: ExactGrader())
    register_grader("normalized", lambda **kw: NormalizedGrader())
    register_grader("numeric", lambda **kw: NumericGrader())
    register_grader("contains", lambda **kw: ContainsGrader())
    register_grader("fuzzy", lambda **kw: FuzzyGrader())
    register_grader("strict", lambda **kw: StrictGrader())
    register_grader("default", lambda **kw: PipelineGrader())
    register_grader("all", lambda **kw: PipelineGrader(matchers=ALL_MATCHERS, grader_name="all"))
    register_grader("llm", lambda **kw: LLMJudgeGrader(model=kw.get("model")))
    register_grader(
        "llm-fallback",
        lambda **kw: LLMFallbackGrader(model=kw.get("model")),
    )


_register_builtins()


def get_grader(name: str, **kwargs: Any) -> Grader:
    """Instantiate a grader by registry name.

    Raises ``ValueError`` if the name is unknown.
    """
    if name not in GRADER_REGISTRY:
        available = ", ".join(sorted(GRADER_REGISTRY.keys()))
        raise ValueError(f"Unknown grader: {name!r}. Available: {available}")
    return GRADER_REGISTRY[name](**kwargs)


def resolve_graders(spec: str, **kwargs: Any) -> list[Grader]:
    """Parse a comma-separated grader spec into a list of ``Grader`` instances.

    Examples::

        resolve_graders("default")           # → [PipelineGrader]
        resolve_graders("exact,llm")         # → [ExactGrader, LLMJudgeGrader]
        resolve_graders("default,llm-fallback", model="gpt-4")
    """
    names = [n.strip() for n in spec.split(",") if n.strip()]
    if not names:
        names = ["default"]
    return [get_grader(n, **kwargs) for n in names]


def list_graders() -> list[str]:
    """Return sorted list of registered grader names."""
    return sorted(GRADER_REGISTRY.keys())


# ============================================================================
# Legacy helpers (backward compatibility)
# ============================================================================

# Keep old names around so existing code that imports them still works.
DEFAULT_GRADERS = DEFAULT_MATCHERS

GRADER_PRESETS = {
    "exact": [("exact", exact_match)],
    "normalized": [("normalized", normalized_match)],
    "numeric": [("numeric", numeric_match)],
    "contains": [("contains", contains_match)],
    "fuzzy": [("fuzzy", fuzzy_match)],
    "strict": [("exact", exact_match), ("normalized", normalized_match)],
    "default": DEFAULT_MATCHERS,
    "all": ALL_MATCHERS,
}


def get_grader_preset(name: str) -> list[tuple[str, Callable[[str, str], bool]]]:
    """Get a legacy grader preset by name."""
    if name not in GRADER_PRESETS:
        available = ", ".join(GRADER_PRESETS.keys())
        raise ValueError(f"Unknown grader preset: {name}. Available: {available}")
    return GRADER_PRESETS[name]


def grade_with_pipeline(
    submission: str,
    expected: str,
    graders: list[tuple[str, Callable[[str, str], bool]]] | None = None,
) -> tuple[bool, str]:
    """Legacy pipeline — try graders in order, return (passed, method)."""
    graders = graders or DEFAULT_MATCHERS
    for name, grader_fn in graders:
        try:
            if grader_fn(submission, expected):
                return True, name
        except Exception:
            continue
    return False, "none"


def create_custom_grader(
    exact: bool = True,
    normalized: bool = True,
    numeric: bool = True,
    numeric_rtol: float = 1e-3,
    contains: bool = False,
    fuzzy: bool = False,
    fuzzy_threshold: float = 0.9,
) -> list[tuple[str, Callable[[str, str], bool]]]:
    """Create a custom legacy grading pipeline."""
    graders: list[tuple[str, Callable[[str, str], bool]]] = []
    if exact:
        graders.append(("exact", exact_match))
    if normalized:
        graders.append(("normalized", normalized_match))
    if numeric:
        graders.append(("numeric", lambda s, e: numeric_match(s, e, rtol=numeric_rtol)))
    if contains:
        graders.append(("contains", contains_match))
    if fuzzy:
        graders.append(("fuzzy", lambda s, e: fuzzy_match(s, e, threshold=fuzzy_threshold)))
    return graders


def llm_judge_match(
    submission: str,
    expected: str,
    model: str | None = None,
) -> bool:
    """Legacy LLM-as-judge function."""
    judge = LLMJudge(model=model)
    passed, _, _ = judge.grade(submission, expected)
    return passed


def grade_with_llm_fallback(
    submission: str,
    expected: str,
    question: str = "",
    graders: list[tuple[str, Callable[[str, str], bool]]] | None = None,
    llm_model: str | None = None,
) -> tuple[bool, str]:
    """Legacy combined grading with LLM fallback."""
    passed, method = grade_with_pipeline(submission, expected, graders)
    if passed:
        return True, method
    judge = LLMJudge(model=llm_model)
    passed, method, _ = judge.grade(submission, expected, question)
    return passed, method

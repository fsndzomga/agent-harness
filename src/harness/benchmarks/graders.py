"""Grading utilities - layered grading pipeline."""

import re
import math
from difflib import SequenceMatcher
from typing import Callable


def exact_match(submission: str, expected: str) -> bool:
    """Exact string match after strip."""
    return submission.strip() == expected.strip()


def normalized_match(submission: str, expected: str) -> bool:
    """Match after normalizing whitespace and case."""
    def normalize(s: str) -> str:
        # Lowercase, collapse whitespace, strip
        return re.sub(r'\s+', ' ', s.lower().strip())
    return normalize(submission) == normalize(expected)


def numeric_match(
    submission: str,
    expected: str,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> bool:
    """
    Match numeric values with tolerance.
    
    Extracts numbers from both strings and compares the last number found.
    This handles cases like "The answer is 42" vs "42" or "2.15" vs "2.1498".
    """
    try:
        # Extract all numbers (including negative and decimal)
        sub_nums = re.findall(r'-?\d+\.?\d*', submission)
        exp_nums = re.findall(r'-?\d+\.?\d*', expected)
        
        if not sub_nums or not exp_nums:
            return False
        
        # Compare last numbers (usually the final answer)
        sub_val = float(sub_nums[-1])
        exp_val = float(exp_nums[-1])
        
        return math.isclose(sub_val, exp_val, rel_tol=rtol, abs_tol=atol)
    except (ValueError, IndexError):
        return False


def fuzzy_match(submission: str, expected: str, threshold: float = 0.9) -> bool:
    """Fuzzy string match using sequence similarity."""
    # Normalize both strings
    sub_norm = re.sub(r'\s+', ' ', submission.lower().strip())
    exp_norm = re.sub(r'\s+', ' ', expected.lower().strip())
    
    ratio = SequenceMatcher(None, sub_norm, exp_norm).ratio()
    return ratio >= threshold


def contains_match(submission: str, expected: str) -> bool:
    """Check if submission contains the expected answer."""
    sub_norm = submission.lower().strip()
    exp_norm = expected.lower().strip()
    return exp_norm in sub_norm


# Default grading pipeline
DEFAULT_GRADERS: list[tuple[str, Callable[[str, str], bool]]] = [
    ("exact", exact_match),
    ("normalized", normalized_match),
    ("numeric", numeric_match),
    ("contains", contains_match),
    ("fuzzy", fuzzy_match),
]


def grade_with_pipeline(
    submission: str,
    expected: str,
    graders: list[tuple[str, Callable[[str, str], bool]]] | None = None,
) -> tuple[bool, str]:
    """
    Try graders in order, return (passed, method).
    
    The pipeline tries graders from strictest to most lenient:
    1. exact - exact string match
    2. normalized - case/whitespace normalized
    3. numeric - numeric tolerance
    4. contains - answer contained in submission
    5. fuzzy - fuzzy string match
    
    Returns as soon as any grader passes.
    """
    graders = graders or DEFAULT_GRADERS
    
    for name, grader in graders:
        try:
            if grader(submission, expected):
                return True, name
        except Exception:
            # Grader failed, try next
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
    """Create a custom grading pipeline."""
    graders = []
    
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


# ============================================================================
# Grader Presets
# ============================================================================

GRADER_PRESETS = {
    "exact": [("exact", exact_match)],
    "normalized": [("normalized", normalized_match)],
    "numeric": [("numeric", numeric_match)],
    "contains": [("contains", contains_match)],
    "fuzzy": [("fuzzy", fuzzy_match)],
    "strict": [("exact", exact_match), ("normalized", normalized_match)],
    "default": DEFAULT_GRADERS,
    "all": DEFAULT_GRADERS,
}


def get_grader_preset(name: str) -> list[tuple[str, Callable[[str, str], bool]]]:
    """Get a grader preset by name."""
    if name not in GRADER_PRESETS:
        available = ", ".join(GRADER_PRESETS.keys())
        raise ValueError(f"Unknown grader preset: {name}. Available: {available}")
    return GRADER_PRESETS[name]


# ============================================================================
# LLM-as-Judge Grader
# ============================================================================

LLM_JUDGE_PROMPT = """You are evaluating whether a submission correctly answers a question.

Question context may be provided. Your task is to determine if the submission is correct.

Expected answer: {expected}
Submission: {submission}

Consider:
1. The submission may be phrased differently but still correct
2. Numeric answers may have minor rounding differences
3. The submission may include extra explanation - focus on the core answer
4. Be lenient with formatting differences (quotes, punctuation, capitalization)

Is the submission correct? Reply with ONLY "CORRECT" or "INCORRECT" (no explanation)."""


class LLMJudge:
    """LLM-as-judge grader for semantic equivalence."""
    
    def __init__(self, model: str | None = None):
        self.model = model
        self._provider = None
    
    @property
    def provider(self):
        if self._provider is None:
            from ..providers import get_provider
            self._provider = get_provider(model=self.model)
        return self._provider
    
    def grade(self, submission: str, expected: str, question: str = "") -> tuple[bool, str, str]:
        """
        Grade using LLM.
        
        Returns: (passed, method, reasoning)
        """
        from ..providers.base import Message, CompletionRequest
        
        prompt = LLM_JUDGE_PROMPT.format(
            expected=expected,
            submission=submission,
        )
        
        if question:
            prompt = f"Question: {question}\n\n{prompt}"
        
        try:
            response = self.provider.complete(CompletionRequest(
                messages=[Message(role="user", content=prompt)],
                max_tokens=10,
                temperature=0.0,
            ))
            
            answer = response.message.content.strip().upper()
            passed = "CORRECT" in answer
            return passed, "llm_judge", answer
            
        except Exception as e:
            # LLM failed - return inconclusive
            return False, "llm_judge_error", str(e)


def llm_judge_match(
    submission: str,
    expected: str,
    model: str | None = None,
) -> bool:
    """LLM-as-judge grader function."""
    judge = LLMJudge(model=model)
    passed, _, _ = judge.grade(submission, expected)
    return passed


# ============================================================================
# Combined Grading with LLM fallback
# ============================================================================

def grade_with_llm_fallback(
    submission: str,
    expected: str,
    question: str = "",
    graders: list[tuple[str, Callable[[str, str], bool]]] | None = None,
    llm_model: str | None = None,
) -> tuple[bool, str]:
    """
    Try deterministic graders first, fall back to LLM-as-judge.
    
    This is the recommended approach:
    1. Fast deterministic checks (exact, normalized, numeric)
    2. LLM judge for uncertain cases
    
    Returns: (passed, method)
    """
    # Try deterministic graders first
    passed, method = grade_with_pipeline(submission, expected, graders)
    
    if passed:
        return True, method
    
    # Fall back to LLM judge
    judge = LLMJudge(model=llm_model)
    passed, method, _ = judge.grade(submission, expected, question)
    
    return passed, method

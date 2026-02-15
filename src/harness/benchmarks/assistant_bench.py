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

import math
import re
from collections import Counter
from typing import Any

from ..protocol import Task
from .base import Benchmark, ExecutionMode, ExecutionContext, GradeResult
from .graders import exact_match, normalized_match, numeric_match, fuzzy_match


# ── Answer-type detection ────────────────────────────────────────────

def _is_numeric(text: str) -> bool:
    """Return ``True`` if *text* can be interpreted as a single number."""
    try:
        float(text.strip().replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def _parse_number(text: str) -> float:
    """Parse a numeric string (handles commas like ``1,010,000``)."""
    return float(text.strip().replace(",", ""))


def _is_list_answer(text: str) -> bool:
    """Return ``True`` if the answer contains multiple newline-separated items."""
    return "\n" in text.strip()


# ── F1 helpers ───────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip, split on word boundaries."""
    return re.findall(r"\w+", text.lower())


def _token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 between two strings."""
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _list_f1(prediction_items: list[str], gold_items: list[str]) -> float:
    """Compute set-level F1 for list answers.

    Each predicted item is matched to its best gold item via token-F1.
    Items matched with F1 >= 0.5 count as a hit.
    """
    if not gold_items and not prediction_items:
        return 1.0
    if not gold_items or not prediction_items:
        return 0.0

    THRESHOLD = 0.5
    matched_gold: set[int] = set()
    hits = 0

    for pred in prediction_items:
        best_score = 0.0
        best_idx = -1
        for idx, gold in enumerate(gold_items):
            if idx in matched_gold:
                continue
            score = _token_f1(pred, gold)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_score >= THRESHOLD and best_idx >= 0:
            matched_gold.add(best_idx)
            hits += 1

    precision = hits / len(prediction_items)
    recall = hits / len(gold_items)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _numeric_closeness(prediction: float, gold: float) -> float:
    """Return a score in [0, 1] measuring how close two numbers are.

    Uses relative error: ``score = max(0, 1 - |pred - gold| / max(|gold|, 1e-9))``
    Returns 1.0 for an exact match and 0.0 when the error is >= 100 %.
    """
    denom = max(abs(gold), 1e-9)
    return max(0.0, 1.0 - abs(prediction - gold) / denom)


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

        sub = submission.strip()
        exp = expected.strip()

        # ── 1. Determine answer type & compute F1 score ──────────
        if _is_numeric(exp):
            f1, method = self._grade_numeric(sub, exp)
        elif _is_list_answer(exp):
            f1, method = self._grade_list(sub, exp)
        else:
            f1, method = self._grade_string(sub, exp)

        passed = f1 >= self.pass_threshold

        return GradeResult(
            task_id=task_id,
            passed=passed,
            score=round(f1, 4),
            expected=expected,
            actual=submission,
            method=method,
            details={
                "f1": round(f1, 4),
                "answer_type": (
                    "numeric" if _is_numeric(exp) else
                    "list" if _is_list_answer(exp) else
                    "string"
                ),
                **(self._metadata.get(task_id, {})),
            },
        )

    def get_num_tasks(self) -> int:
        """Return number of tasks without loading all task objects."""
        return len(self.get_tasks())

    # ------------------------------------------------------------------
    # Grading helpers
    # ------------------------------------------------------------------

    def _grade_string(self, sub: str, exp: str) -> tuple[float, str]:
        """Grade a plain-string answer.  Returns ``(f1, method)``."""
        if exact_match(sub, exp):
            return 1.0, "exact"
        if normalized_match(sub, exp):
            return 1.0, "normalized"
        if fuzzy_match(sub, exp, threshold=0.95):
            return 1.0, "fuzzy"

        f1 = _token_f1(sub, exp)
        return f1, "token_f1"

    def _grade_numeric(self, sub: str, exp: str) -> tuple[float, str]:
        """Grade a numeric answer.  Returns ``(score, method)``."""
        if exact_match(sub, exp):
            return 1.0, "exact"
        if normalized_match(sub, exp):
            return 1.0, "normalized"

        # Try to extract a number from the submission
        try:
            gold_val = _parse_number(exp)
        except ValueError:
            return 0.0, "numeric_parse_error"

        # Try the raw submission first, then extract last number
        pred_val: float | None = None
        try:
            pred_val = _parse_number(sub)
        except ValueError:
            nums = re.findall(r"-?\d[\d,]*\.?\d*", sub)
            if nums:
                try:
                    pred_val = _parse_number(nums[-1])
                except ValueError:
                    pass

        if pred_val is None:
            return 0.0, "numeric_no_number"

        if math.isclose(pred_val, gold_val, rel_tol=1e-3, abs_tol=1e-6):
            return 1.0, "numeric_exact"

        score = _numeric_closeness(pred_val, gold_val)
        return score, "numeric_close"

    def _grade_list(self, sub: str, exp: str) -> tuple[float, str]:
        """Grade a list answer (newline-separated items). Returns ``(f1, method)``."""
        gold_items = [item.strip() for item in exp.split("\n") if item.strip()]
        pred_items = [item.strip() for item in sub.split("\n") if item.strip()]

        if not pred_items:
            return 0.0, "list_empty"

        # Check if both sides are identical after normalization
        if exact_match(sub, exp):
            return 1.0, "exact"
        if normalized_match(sub, exp):
            return 1.0, "normalized"

        f1 = _list_f1(pred_items, gold_items)
        return f1, "list_f1"

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

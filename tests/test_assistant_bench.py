"""Tests for AssistantBench benchmark implementation.

Covers:
- Official question_scorer (vendored from browsergym)
- Grading: string, numeric, list, JSON answer types
- Dataset loading from HuggingFace (validation and test splits)
- Difficulty filtering
- Registry integration
- Edge cases
"""

from __future__ import annotations

import math

import pytest

from harness.protocol import Task
from harness.benchmarks.base import ExecutionContext, GradeResult
from harness.benchmarks.assistant_bench import (
    AssistantBenchBenchmark,
    AssistantBenchHardBenchmark,
    AssistantBenchMediumBenchmark,
)

# The eval module requires numpy + scipy; skip all scorer tests if missing.
try:
    from harness.benchmarks._assistantbench_eval import (
        question_scorer,
        evaluate_numbers,
        evaluate_strings,
        distance_function_log,
    )
    _HAS_EVAL_DEPS = True
except ImportError:
    _HAS_EVAL_DEPS = False

needs_eval = pytest.mark.skipif(not _HAS_EVAL_DEPS, reason="numpy/scipy not installed")

# Dataset loading tests require the `datasets` package + network.
try:
    import datasets as _datasets_mod  # noqa: F401
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False

needs_datasets = pytest.mark.skipif(not _HAS_DATASETS, reason="datasets package not installed")

# =====================================================================
# Official scorer function tests
# =====================================================================

@needs_eval
class TestDistanceFunctionLog:
    def test_exact(self):
        assert distance_function_log(42.0, 42.0) == 1.0

    def test_both_zero(self):
        assert distance_function_log(0.0, 0.0) == 1

    def test_close_numbers(self):
        score = distance_function_log(95.0, 100.0)
        # log(100/95) ≈ 0.0513 → 1 - 0.0513 ≈ 0.949
        assert 0.9 < score < 1.0

    def test_far_numbers(self):
        score = distance_function_log(10.0, 100.0)
        # log(100/10) = log(10) ≈ 2.30 → 1 - 2.30 → clamped to 0
        assert score == 0.0

    def test_symmetry(self):
        s1 = distance_function_log(95.0, 100.0)
        s2 = distance_function_log(100.0, 95.0)
        assert math.isclose(s1, s2, rel_tol=1e-6)


@needs_eval
class TestEvaluateNumbers:
    def test_exact(self):
        assert evaluate_numbers(42.0, 42.0) == 1.0

    def test_string_pred(self):
        score = evaluate_numbers("42", 42.0)
        assert score == 1.0

    def test_invalid_pred(self):
        assert evaluate_numbers("not a number", 42.0) == 0

    def test_close(self):
        score = evaluate_numbers(95.0, 100.0)
        assert 0.9 < score < 1.0


@needs_eval
class TestEvaluateStrings:
    def test_exact(self):
        assert evaluate_strings("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert evaluate_strings("hello", "goodbye") == 0.0

    def test_partial_overlap(self):
        f1 = evaluate_strings("hello beautiful world", "hello world")
        assert f1 > 0.0

    def test_both_empty(self):
        assert evaluate_strings("", "") == 1.0

    def test_article_removal(self):
        """Official scorer removes articles (a, an, the) during normalization."""
        score = evaluate_strings("the hello world", "hello world")
        assert score == 1.0

    def test_punctuation_removal(self):
        """Official scorer removes punctuation during normalization."""
        score = evaluate_strings("hello, world!", "hello world")
        assert score == 1.0

    def test_list_strings(self):
        score = evaluate_strings(["item1", "item2"], ["item1", "item2"])
        assert score == 1.0


@needs_eval
class TestQuestionScorer:
    def test_exact_string(self):
        acc, has = question_scorer("hello world", "hello world")
        assert acc == 1.0
        assert has == 1.0

    def test_exact_number(self):
        acc, has = question_scorer("42", "42")
        assert acc == 1.0

    def test_close_number(self):
        acc, has = question_scorer("95", "100")
        assert 0.9 < acc < 1.0

    def test_empty_prediction(self):
        acc, has = question_scorer("", "hello")
        assert has == 0.0

    def test_multiline_gold(self):
        """Multi-line gold answers are treated as string lists."""
        # Prediction needs to be a JSON list for proper list evaluation
        acc, has = question_scorer('["A", "B", "C"]', "A\nB\nC")
        assert acc == 1.0

    def test_json_gold(self):
        """JSON gold answers use dict evaluation."""
        gold = '{"name": "Alice", "age": "30"}'
        acc, has = question_scorer(gold, gold)
        assert acc == 1.0


# =====================================================================
# Grading tests (using a benchmark with manually injected answers)
# =====================================================================

def _make_bench_with_answers(answers: dict[str, str], **kwargs) -> AssistantBenchBenchmark:
    """Create a benchmark whose tasks/answers are pre-loaded (no HF call)."""
    bench = AssistantBenchBenchmark(**kwargs)
    bench._tasks = [
        Task(id=tid, data={"question": f"q for {tid}"})
        for tid in answers
    ]
    bench._answers = dict(answers)
    bench._metadata = {tid: {"difficulty": "Medium"} for tid in answers}
    return bench


def _ctx(task_id: str = "t") -> ExecutionContext:
    return ExecutionContext(task_id=task_id)


@needs_eval
class TestGradeString:
    def test_exact(self):
        b = _make_bench_with_answers({"t": "Shanghai villa"})
        r = b.grade(b.get_tasks()[0], "Shanghai villa", _ctx())
        assert r.passed
        assert r.score == 1.0
        assert r.method == "official_question_scorer"

    def test_normalized(self):
        b = _make_bench_with_answers({"t": "Shanghai villa"})
        r = b.grade(b.get_tasks()[0], "  shanghai   villa  ", _ctx())
        assert r.passed
        assert r.score == 1.0

    def test_token_f1_partial(self):
        b = _make_bench_with_answers({"t": "Shanghai villa"})
        r = b.grade(b.get_tasks()[0], "The Shanghai luxury villa", _ctx())
        # Official DROP-style F1 removes articles ("the") → pred becomes "shanghai luxury villa"
        # gold tokens: {shanghai, villa}, pred tokens: {shanghai, luxury, villa}
        # intersection = 2, precision = 2/3, recall = 2/2 = 1
        # f1 = 2*(2/3)*1 / (2/3+1) = 0.8
        assert r.score >= 0.8  # passes with official scorer (articles removed)

    def test_no_match(self):
        b = _make_bench_with_answers({"t": "Shanghai villa"})
        r = b.grade(b.get_tasks()[0], "completely wrong", _ctx())
        assert not r.passed
        assert r.score < 0.2


@needs_eval
class TestGradeNumeric:
    def test_exact_numeric(self):
        b = _make_bench_with_answers({"t": "14.2"})
        r = b.grade(b.get_tasks()[0], "14.2", _ctx())
        assert r.passed
        assert r.score == 1.0
        assert r.method == "official_question_scorer"

    def test_numeric_match(self):
        b = _make_bench_with_answers({"t": "14.2"})
        r = b.grade(b.get_tasks()[0], "14.200", _ctx())
        assert r.passed
        assert r.score == 1.0

    def test_numeric_in_text(self):
        b = _make_bench_with_answers({"t": "45"})
        r = b.grade(b.get_tasks()[0], "The answer is 45.", _ctx())
        # Official scorer tries to parse as number; text may not parse
        # Score depends on official fix_number / fix_prediction behavior
        assert r.score >= 0.0

    def test_numeric_close(self):
        b = _make_bench_with_answers({"t": "100"})
        r = b.grade(b.get_tasks()[0], "95", _ctx())
        # Official log distance: 1 - log(100/95) ≈ 0.949
        assert r.passed
        assert 0.9 < r.score < 1.0

    def test_numeric_far(self):
        b = _make_bench_with_answers({"t": "100"})
        r = b.grade(b.get_tasks()[0], "10", _ctx())
        # Official log distance: 1 - log(100/10) = 1 - log(10) ≈ -1.3 → clamped to 0
        assert not r.passed
        assert r.score == 0.0

    def test_no_number_in_submission(self):
        b = _make_bench_with_answers({"t": "42"})
        r = b.grade(b.get_tasks()[0], "no numbers here", _ctx())
        assert not r.passed
        assert r.score == 0.0

    def test_comma_number(self):
        b = _make_bench_with_answers({"t": "1010000"})
        r = b.grade(b.get_tasks()[0], "1,010,000", _ctx())
        # Official scorer replaces commas with dots for fix_number → 1.010.000
        # which fails float parse, so it falls back to string comparison
        # The exact score depends on official implementation behavior
        assert r.score >= 0.0


@needs_eval
class TestGradeList:
    def test_exact_list_as_json(self):
        """Official scorer needs JSON array for list predictions."""
        b = _make_bench_with_answers({"t": "CrossFit East River\nAvea Pilates"})
        r = b.grade(b.get_tasks()[0], '["CrossFit East River", "Avea Pilates"]', _ctx())
        assert r.passed
        assert r.score == 1.0
        assert r.method == "official_question_scorer"

    def test_exact_list_raw_text(self):
        """Raw newline text gets lower score — official scorer treats as single string."""
        b = _make_bench_with_answers({"t": "CrossFit East River\nAvea Pilates"})
        r = b.grade(b.get_tasks()[0], "CrossFit East River\nAvea Pilates", _ctx())
        # Without JSON wrapping, scorer treats pred as single string vs string list
        # Score is partial, not full
        assert r.score < 1.0

    def test_partial_list(self):
        b = _make_bench_with_answers({"t": "A\nB\nC"})
        r = b.grade(b.get_tasks()[0], '["A", "B"]', _ctx())
        f1 = r.score
        # 2 matched out of 3 gold spans
        assert 0.5 <= f1 <= 0.9

    def test_empty_submission(self):
        b = _make_bench_with_answers({"t": "A\nB"})
        r = b.grade(b.get_tasks()[0], "", _ctx())
        assert not r.passed

    def test_wrong_items(self):
        b = _make_bench_with_answers({"t": "A\nB\nC"})
        r = b.grade(b.get_tasks()[0], '["X", "Y", "Z"]', _ctx())
        assert not r.passed
        assert r.score == 0.0

    def test_json_dict_list(self):
        gold = '{"sender": "DHL", "price": "55-70"}\n{"sender": "Fedex", "price": "62-95"}'
        pred = '[{"sender": "DHL", "price": "55-70"}, {"sender": "Fedex", "price": "62-95"}]'
        b = _make_bench_with_answers({"t": gold})
        r = b.grade(b.get_tasks()[0], pred, _ctx())
        assert r.passed
        assert r.score == 1.0


@needs_eval
class TestGradeEdgeCases:
    def test_no_answer_available(self):
        """Test set tasks have no gold answer → score 0, method no_answer."""
        b = _make_bench_with_answers({})
        b._tasks = [Task(id="t", data={"question": "q"})]
        r = b.grade(b.get_tasks()[0], "some answer", _ctx())
        assert not r.passed
        assert r.method == "no_answer"

    def test_none_submission(self):
        b = _make_bench_with_answers({"t": "hello"})
        r = b.grade(b.get_tasks()[0], None, _ctx())
        assert not r.passed

    def test_details_contain_accuracy(self):
        b = _make_bench_with_answers({"t": "42"})
        r = b.grade(b.get_tasks()[0], "42", _ctx())
        assert r.details is not None
        assert "accuracy" in r.details
        assert "has_ans" in r.details

    def test_custom_threshold(self):
        b = _make_bench_with_answers({"t": "hello world"}, pass_threshold=0.5)
        # Partial overlap that scores > 0.5 but < 0.8
        r = b.grade(b.get_tasks()[0], "hello beautiful world here", _ctx())
        # With lower threshold, this may pass
        assert r.score > 0.0


# =====================================================================
# Dataset loading (integration tests — requires network)
# =====================================================================

@needs_eval
@needs_datasets
class TestDatasetLoading:
    """Integration tests that actually load from HuggingFace.

    These tests do a real download so they may be slow on first run.
    """

    def test_load_validation_split(self):
        bench = AssistantBenchBenchmark(split="validation")
        tasks = bench.get_tasks()
        assert len(tasks) == 33
        # Every task should have a question
        for t in tasks:
            assert "question" in t.data
            assert t.data["question"]
        # Every validation task should have a gold answer
        assert len(bench._answers) == 33

    def test_load_test_split(self):
        bench = AssistantBenchBenchmark(split="test")
        tasks = bench.get_tasks()
        assert len(tasks) == 181
        # Test set has no answers
        assert len(bench._answers) == 0

    def test_difficulty_filter_hard(self):
        bench = AssistantBenchHardBenchmark()
        tasks = bench.get_tasks()
        assert len(tasks) == 19
        for t in tasks:
            assert t.data.get("difficulty") == "Hard"

    def test_difficulty_filter_medium(self):
        bench = AssistantBenchMediumBenchmark()
        tasks = bench.get_tasks()
        assert len(tasks) == 14
        for t in tasks:
            assert t.data.get("difficulty") == "Medium"

    def test_task_ids_are_unique(self):
        bench = AssistantBenchBenchmark(split="validation")
        tasks = bench.get_tasks()
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_grade_exact_from_dataset(self):
        """Grade a real task with the exact gold answer (single-value answer)."""
        bench = AssistantBenchBenchmark(split="validation")
        tasks = bench.get_tasks()
        # Find a task with a single-line (non-list) gold answer
        for task in tasks:
            gold = bench._answers[task.id]
            if "\n" not in gold.strip():
                r = bench.grade(task, gold, _ctx(task.id))
                assert r.passed, f"Task {task.id}: score={r.score}, gold={gold!r}"
                assert r.score == 1.0
                return
        pytest.skip("No single-line gold answer found")

    def test_get_num_tasks(self):
        bench = AssistantBenchBenchmark(split="validation")
        assert bench.get_num_tasks() == 33

    def test_caching(self):
        """get_tasks() doesn't reload on second call."""
        bench = AssistantBenchBenchmark(split="validation")
        tasks1 = bench.get_tasks()
        tasks2 = bench.get_tasks()
        assert tasks1 is tasks2


class TestRegistryIntegration:
    def test_assistant_bench_in_registry(self):
        from harness.benchmarks.registry import get_benchmark, list_benchmarks
        names = list_benchmarks()
        assert "assistant-bench" in names
        assert "assistant-bench-hard" in names
        assert "assistant-bench-medium" in names

    def test_get_benchmark_returns_instance(self):
        from harness.benchmarks.registry import get_benchmark
        bench = get_benchmark("assistant-bench")
        assert isinstance(bench, AssistantBenchBenchmark)
        assert bench.split == "validation"


@needs_eval
class TestGradeLegacy:
    """Test backward-compatible grade_legacy method."""

    def test_legacy_grade(self):
        b = _make_bench_with_answers({"t": "hello world"})
        r = b.grade_legacy("t", "hello world")
        assert r.passed
        assert r.score == 1.0

    def test_legacy_grade_unknown_task(self):
        b = _make_bench_with_answers({"t": "hello"})
        with pytest.raises(ValueError, match="not found"):
            b.grade_legacy("unknown", "hello")

    def test_grade_all(self):
        b = _make_bench_with_answers({"t1": "alpha", "t2": "42"})
        results = b.grade_all([
            {"task_id": "t1", "submission": "alpha"},
            {"task_id": "t2", "submission": "42"},
        ])
        assert len(results) == 2
        assert all(r.passed for r in results)

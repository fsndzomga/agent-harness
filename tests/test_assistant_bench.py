"""Tests for AssistantBench benchmark implementation.

Covers:
- Helper functions (_is_numeric, _parse_number, _is_list_answer, _tokenize,
  _token_f1, _list_f1, _numeric_closeness)
- Grading: string, numeric, list answer types
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
    _is_numeric,
    _parse_number,
    _is_list_answer,
    _tokenize,
    _token_f1,
    _list_f1,
    _numeric_closeness,
)


# =====================================================================
# Helper function tests
# =====================================================================

class TestIsNumeric:
    def test_integers(self):
        assert _is_numeric("42")
        assert _is_numeric("0")
        assert _is_numeric("-5")

    def test_floats(self):
        assert _is_numeric("14.2")
        assert _is_numeric("-3.14")
        assert _is_numeric("0.001")

    def test_with_commas(self):
        assert _is_numeric("1,010,000")
        assert _is_numeric("1,000")

    def test_with_whitespace(self):
        assert _is_numeric("  42  ")

    def test_non_numeric(self):
        assert not _is_numeric("hello")
        assert not _is_numeric("42km")
        assert not _is_numeric("")
        assert not _is_numeric("one")
        assert not _is_numeric("1.2.3")


class TestParseNumber:
    def test_plain(self):
        assert _parse_number("42") == 42.0
        assert _parse_number("-5") == -5.0

    def test_float(self):
        assert math.isclose(_parse_number("14.2"), 14.2)

    def test_commas(self):
        assert _parse_number("1,010,000") == 1010000.0

    def test_whitespace(self):
        assert _parse_number("  100  ") == 100.0

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_number("abc")


class TestIsListAnswer:
    def test_single_line(self):
        assert not _is_list_answer("hello")
        assert not _is_list_answer("42")

    def test_multi_line(self):
        assert _is_list_answer("item1\nitem2")
        assert _is_list_answer("a\nb\nc")

    def test_trailing_newline_only(self):
        # A single trailing newline after stripping is not a list
        assert not _is_list_answer("hello\n")
        assert not _is_list_answer("hello\n  ")


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_punctuation(self):
        assert _tokenize("hello, world!") == ["hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []
        assert _tokenize("   ") == []


class TestTokenF1:
    def test_exact(self):
        assert _token_f1("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _token_f1("hello", "goodbye") == 0.0

    def test_partial_overlap(self):
        f1 = _token_f1("hello beautiful world", "hello world")
        # pred has 3 tokens, gold has 2, overlap is 2
        # precision = 2/3, recall = 2/2 = 1
        # f1 = 2 * (2/3) * 1 / (2/3 + 1) = (4/3) / (5/3) = 4/5 = 0.8
        assert math.isclose(f1, 0.8, rel_tol=1e-6)

    def test_both_empty(self):
        assert _token_f1("", "") == 1.0

    def test_one_empty(self):
        assert _token_f1("hello", "") == 0.0
        assert _token_f1("", "hello") == 0.0


class TestListF1:
    def test_exact(self):
        gold = ["item1", "item2", "item3"]
        pred = ["item1", "item2", "item3"]
        assert _list_f1(pred, gold) == 1.0

    def test_partial_match(self):
        gold = ["item1", "item2", "item3"]
        pred = ["item1", "item2"]
        f1 = _list_f1(pred, gold)
        # precision = 2/2 = 1.0, recall = 2/3
        # f1 = 2 * 1 * (2/3) / (1 + 2/3) = (4/3) / (5/3) = 0.8
        assert math.isclose(f1, 0.8, rel_tol=1e-6)

    def test_no_match(self):
        gold = ["alpha", "beta"]
        pred = ["gamma", "delta"]
        assert _list_f1(pred, gold) == 0.0

    def test_both_empty(self):
        assert _list_f1([], []) == 1.0

    def test_one_empty(self):
        assert _list_f1([], ["a"]) == 0.0
        assert _list_f1(["a"], []) == 0.0

    def test_extra_predictions(self):
        gold = ["item1"]
        pred = ["item1", "item2", "item3"]
        f1 = _list_f1(pred, gold)
        # precision = 1/3, recall = 1/1 = 1
        # f1 = 2 * (1/3) * 1 / (1/3 + 1) = (2/3) / (4/3) = 0.5
        assert math.isclose(f1, 0.5, rel_tol=1e-6)


class TestNumericCloseness:
    def test_exact(self):
        assert _numeric_closeness(42.0, 42.0) == 1.0

    def test_off_by_half(self):
        # |50 - 100| / 100 = 0.5 → 1 - 0.5 = 0.5
        assert math.isclose(_numeric_closeness(50.0, 100.0), 0.5)

    def test_off_by_100_percent(self):
        # |0 - 100| / 100 = 1.0 → 1 - 1 = 0.0
        assert _numeric_closeness(0.0, 100.0) == 0.0

    def test_overshot(self):
        # |200 - 100| / 100 = 1.0 → 1 - 1 = 0.0
        assert _numeric_closeness(200.0, 100.0) == 0.0


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


class TestGradeString:
    def test_exact(self):
        b = _make_bench_with_answers({"t": "Shanghai villa"})
        r = b.grade(b.get_tasks()[0], "Shanghai villa", _ctx())
        assert r.passed
        assert r.score == 1.0
        assert r.method == "exact"

    def test_normalized(self):
        b = _make_bench_with_answers({"t": "Shanghai villa"})
        r = b.grade(b.get_tasks()[0], "  shanghai   villa  ", _ctx())
        assert r.passed
        assert r.score == 1.0
        assert r.method == "normalized"

    def test_token_f1_partial(self):
        b = _make_bench_with_answers({"t": "Shanghai villa"})
        r = b.grade(b.get_tasks()[0], "The Shanghai luxury villa", _ctx())
        # overlap tokens: shanghai, villa (2)
        # pred tokens: the, shanghai, luxury, villa (4)
        # gold tokens: shanghai, villa (2)
        # precision = 2/4 = 0.5, recall = 2/2 = 1.0
        # f1 = 2 * 0.5 * 1 / 1.5 = 0.667
        assert not r.passed  # below 0.8 threshold
        assert r.method == "token_f1"
        assert 0.6 < r.score < 0.7

    def test_no_match(self):
        b = _make_bench_with_answers({"t": "Shanghai villa"})
        r = b.grade(b.get_tasks()[0], "completely wrong", _ctx())
        assert not r.passed
        assert r.score < 0.2


class TestGradeNumeric:
    def test_exact_numeric(self):
        b = _make_bench_with_answers({"t": "14.2"})
        r = b.grade(b.get_tasks()[0], "14.2", _ctx())
        assert r.passed
        assert r.score == 1.0
        assert r.method == "exact"

    def test_numeric_match(self):
        b = _make_bench_with_answers({"t": "14.2"})
        r = b.grade(b.get_tasks()[0], "14.200", _ctx())
        assert r.passed
        assert r.score == 1.0
        assert r.method == "numeric_exact"

    def test_numeric_in_text(self):
        b = _make_bench_with_answers({"t": "45"})
        r = b.grade(b.get_tasks()[0], "The answer is 45.", _ctx())
        assert r.passed
        assert r.score == 1.0

    def test_numeric_close(self):
        b = _make_bench_with_answers({"t": "100"})
        r = b.grade(b.get_tasks()[0], "95", _ctx())
        # closeness = 1 - 5/100 = 0.95
        assert r.passed
        assert r.method == "numeric_close"
        assert math.isclose(r.score, 0.95, rel_tol=1e-2)

    def test_numeric_far(self):
        b = _make_bench_with_answers({"t": "100"})
        r = b.grade(b.get_tasks()[0], "10", _ctx())
        # closeness = 1 - 90/100 = 0.1
        assert not r.passed
        assert r.score < 0.2

    def test_no_number_in_submission(self):
        b = _make_bench_with_answers({"t": "42"})
        r = b.grade(b.get_tasks()[0], "no numbers here", _ctx())
        assert not r.passed
        assert r.score == 0.0
        assert r.method == "numeric_no_number"

    def test_comma_number(self):
        b = _make_bench_with_answers({"t": "1010000"})
        r = b.grade(b.get_tasks()[0], "1,010,000", _ctx())
        assert r.passed
        assert r.score == 1.0


class TestGradeList:
    def test_exact_list(self):
        b = _make_bench_with_answers({"t": "CrossFit East River\nAvea Pilates"})
        r = b.grade(b.get_tasks()[0], "CrossFit East River\nAvea Pilates", _ctx())
        assert r.passed
        assert r.score == 1.0
        assert r.method == "exact"

    def test_partial_list(self):
        b = _make_bench_with_answers({"t": "A\nB\nC"})
        r = b.grade(b.get_tasks()[0], "A\nB", _ctx())
        f1 = r.score
        # 2 correct out of 3 gold → recall = 2/3, precision = 2/2 = 1
        # f1 = 2 * 1 * (2/3) / (1 + 2/3) = 0.8
        assert math.isclose(f1, 0.8, rel_tol=1e-2)
        assert r.passed  # exactly at threshold 0.8

    def test_empty_submission(self):
        b = _make_bench_with_answers({"t": "A\nB"})
        r = b.grade(b.get_tasks()[0], "", _ctx())
        assert not r.passed
        assert r.score == 0.0

    def test_wrong_items(self):
        b = _make_bench_with_answers({"t": "A\nB\nC"})
        r = b.grade(b.get_tasks()[0], "X\nY\nZ", _ctx())
        assert not r.passed
        assert r.score == 0.0

    def test_json_list(self):
        gold = '{"sender": "DHL", "price": "55-70"}\n{"sender": "Fedex", "price": "62-95"}'
        b = _make_bench_with_answers({"t": gold})
        r = b.grade(b.get_tasks()[0], gold, _ctx())
        assert r.passed
        assert r.score == 1.0


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

    def test_details_contain_answer_type(self):
        b = _make_bench_with_answers({"t": "42"})
        r = b.grade(b.get_tasks()[0], "42", _ctx())
        assert r.details is not None
        assert r.details["answer_type"] == "numeric"

    def test_custom_threshold(self):
        b = _make_bench_with_answers({"t": "hello world"}, pass_threshold=0.5)
        # Partial overlap that scores > 0.5 but < 0.8
        r = b.grade(b.get_tasks()[0], "hello beautiful world here", _ctx())
        # With lower threshold, this may pass
        assert r.score > 0.0


# =====================================================================
# Dataset loading (integration tests — requires network)
# =====================================================================

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
        """Grade a real task with the exact gold answer."""
        bench = AssistantBenchBenchmark(split="validation")
        tasks = bench.get_tasks()
        task = tasks[0]
        gold = bench._answers[task.id]
        r = bench.grade(task, gold, _ctx(task.id))
        assert r.passed
        assert r.score == 1.0

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

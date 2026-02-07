"""Tests for the multi-grader system.

Covers:
  - Grader ABC contract (can't instantiate without name/grade_one)
  - All built-in graders produce correct GradeResult
  - Registry: get_grader, resolve_graders, list_graders
  - run_graders orchestration with multi-grader
  - grades.json I/O (save/load, merge, backward compat with old list format)
  - run.json scores patching (patch_run_json_scores, merge)
  - PipelineGrader matches ordering
"""

import json
import tempfile
from pathlib import Path

import pytest

from harness.benchmarks.base import GradeResult
from harness.benchmarks.graders import (
    # ABC
    Grader,
    # Concrete graders
    ExactGrader,
    NormalizedGrader,
    NumericGrader,
    ContainsGrader,
    FuzzyGrader,
    StrictGrader,
    PipelineGrader,
    # Registry
    get_grader,
    resolve_graders,
    list_graders,
    GRADER_REGISTRY,
    # Legacy helpers
    exact_match,
    normalized_match,
    numeric_match,
    fuzzy_match,
    contains_match,
    grade_with_pipeline,
    get_grader_preset,
)
from harness.grading import (
    run_graders,
    save_grades,
    load_grades,
    load_grades_raw,
    patch_run_json_scores,
)


# ============================================================================
# Grader ABC Contract
# ============================================================================

class TestGraderABC:
    """Verify the Grader ABC prevents incomplete subclasses."""

    def test_cannot_instantiate_bare_grader(self):
        with pytest.raises(TypeError):
            Grader()  # type: ignore[abstract]

    def test_missing_name_raises(self):
        class NoName(Grader):
            def grade_one(self, submission, expected, **kw):
                return True, "x"

        with pytest.raises(TypeError):
            NoName()

    def test_missing_grade_one_raises(self):
        class NoGradeOne(Grader):
            @property
            def name(self):
                return "bogus"

        with pytest.raises(TypeError):
            NoGradeOne()

    def test_complete_subclass_works(self):
        class MyGrader(Grader):
            @property
            def name(self):
                return "mine"

            def grade_one(self, submission, expected, **kw):
                return submission == expected, "mine"

        g = MyGrader()
        assert g.name == "mine"
        assert g.grade_one("a", "a") == (True, "mine")
        assert g.grade_one("a", "b") == (False, "mine")


# ============================================================================
# Built-in Graders
# ============================================================================

class TestBuiltinGraders:

    def test_exact(self):
        g = ExactGrader()
        assert g.name == "exact"
        assert g.grade_one("42", "42") == (True, "exact")
        assert g.grade_one(" 42 ", " 42 ") == (True, "exact")
        assert g.grade_one("42", "43") == (False, "exact")

    def test_normalized(self):
        g = NormalizedGrader()
        assert g.name == "normalized"
        assert g.grade_one("Hello World", "hello  world") == (True, "normalized")
        assert g.grade_one("abc", "xyz") == (False, "normalized")

    def test_numeric(self):
        g = NumericGrader()
        assert g.name == "numeric"
        assert g.grade_one("2.15", "2.1498")[0] is True
        assert g.grade_one("no num", "42")[0] is False

    def test_contains(self):
        g = ContainsGrader()
        assert g.name == "contains"
        assert g.grade_one("The answer is 42", "42") == (True, "contains")
        assert g.grade_one("nope", "42") == (False, "contains")

    def test_fuzzy(self):
        g = FuzzyGrader(threshold=0.8)
        assert g.name == "fuzzy"
        assert g.grade_one("hello world", "hello  world")[0] is True
        assert g.grade_one("abc", "xyz")[0] is False

    def test_strict(self):
        g = StrictGrader()
        assert g.name == "strict"
        assert g.grade_one("42", "42") == (True, "exact")
        assert g.grade_one("Hello", "hello") == (True, "normalized")
        assert g.grade_one("abc", "xyz") == (False, "none")

    def test_pipeline_default(self):
        g = PipelineGrader()
        assert g.name == "default"
        # Exact hit
        assert g.grade_one("42", "42") == (True, "exact")
        # Normalized hit
        assert g.grade_one("HELLO", "hello")[0] is True

    def test_graderesult_from_grade(self):
        g = ExactGrader()
        result = g.grade("t1", "42", "42")
        assert isinstance(result, GradeResult)
        assert result.task_id == "t1"
        assert result.passed is True
        assert result.score == 1.0
        assert result.method == "exact"

    def test_grade_batch(self):
        g = ExactGrader()
        batch = [
            {"task_id": "a", "submission": "42", "expected": "42"},
            {"task_id": "b", "submission": "43", "expected": "42"},
        ]
        results = g.grade_batch(batch)
        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False


# ============================================================================
# Registry
# ============================================================================

class TestGraderRegistry:

    def test_list_graders(self):
        names = list_graders()
        assert "exact" in names
        assert "default" in names
        assert "llm" in names

    def test_get_grader_known(self):
        g = get_grader("exact")
        assert isinstance(g, ExactGrader)

    def test_get_grader_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown grader"):
            get_grader("no_such_grader")

    def test_resolve_single(self):
        graders = resolve_graders("exact")
        assert len(graders) == 1
        assert graders[0].name == "exact"

    def test_resolve_multiple(self):
        graders = resolve_graders("exact,normalized,strict")
        assert len(graders) == 3
        names = [g.name for g in graders]
        assert names == ["exact", "normalized", "strict"]

    def test_resolve_empty_defaults(self):
        graders = resolve_graders("")
        assert len(graders) == 1
        assert graders[0].name == "default"


# ============================================================================
# run_graders orchestration
# ============================================================================

class TestRunGraders:

    def test_single_grader(self):
        graders = [ExactGrader()]
        subs = [
            {"task_id": "t1", "submission": "42", "expected": "42"},
            {"task_id": "t2", "submission": "43", "expected": "42"},
        ]
        results = run_graders(graders, subs)
        assert "exact" in results
        assert len(results["exact"]) == 2
        assert results["exact"][0].passed is True
        assert results["exact"][1].passed is False

    def test_multi_grader(self):
        graders = [ExactGrader(), ContainsGrader()]
        subs = [
            {"task_id": "t1", "submission": "The answer is 42", "expected": "42"},
        ]
        results = run_graders(graders, subs)
        assert "exact" in results
        assert "contains" in results
        # Exact fails (full string != "42"), Contains passes
        assert results["exact"][0].passed is False
        assert results["contains"][0].passed is True

    def test_failed_tasks_counted(self):
        graders = [ExactGrader()]
        subs = [{"task_id": "t1", "submission": "42", "expected": "42"}]
        failed = [{"task_id": "t2", "expected": "99", "error": "timeout"}]
        results = run_graders(graders, subs, failed)
        assert len(results["exact"]) == 2
        assert results["exact"][1].passed is False
        assert results["exact"][1].method == "failed"


# ============================================================================
# grades.json I/O
# ============================================================================

class TestGradesIO:

    def _make_path(self, tmp_path: Path) -> Path:
        return tmp_path / "grades.json"

    def test_save_and_load(self, tmp_path):
        gpath = self._make_path(tmp_path)
        multi = {
            "exact": [
                GradeResult("t1", True, 1.0, "42", "42", "exact"),
                GradeResult("t2", False, 0.0, "99", "nope", "exact"),
            ],
        }
        save_grades(gpath, multi)
        loaded = load_grades(gpath)
        assert "exact" in loaded
        assert len(loaded["exact"]) == 2
        assert loaded["exact"][0].passed is True

    def test_save_multi_grader(self, tmp_path):
        gpath = self._make_path(tmp_path)
        multi = {
            "exact": [GradeResult("t1", True, 1.0, "42", "42", "exact")],
            "contains": [GradeResult("t1", True, 1.0, "42", "ans: 42", "contains")],
        }
        save_grades(gpath, multi)
        raw = load_grades_raw(gpath)
        assert set(raw.keys()) == {"exact", "contains"}

    def test_merge(self, tmp_path):
        gpath = self._make_path(tmp_path)
        # Save first grader
        save_grades(gpath, {
            "exact": [GradeResult("t1", True, 1.0, "42", "42", "exact")],
        })
        # Merge second grader
        save_grades(gpath, {
            "contains": [GradeResult("t1", True, 1.0, "42", "has 42", "contains")],
        }, merge=True)

        raw = load_grades_raw(gpath)
        assert set(raw.keys()) == {"exact", "contains"}

    def test_backward_compat_old_list_format(self, tmp_path):
        gpath = self._make_path(tmp_path)
        # Old format: plain list
        gpath.write_text(json.dumps([
            {"task_id": "t1", "passed": True, "score": 1.0,
             "expected": "42", "actual": "42", "method": "exact"},
        ]))
        raw = load_grades_raw(gpath)
        assert "default" in raw
        assert len(raw["default"]) == 1


# ============================================================================
# run.json scores patching
# ============================================================================

class TestPatchRunJsonScores:

    def test_patch_adds_scores(self, tmp_path):
        run_json = tmp_path / "run.json"
        run_json.write_text(json.dumps({"run_id": "test", "score": 0, "scores": {}}))

        multi = {
            "exact": [
                GradeResult("t1", True, 1.0, "42", "42", "exact"),
                GradeResult("t2", False, 0.0, "99", "no", "exact"),
            ],
        }
        data = patch_run_json_scores(run_json, multi)
        assert "exact" in data["scores"]
        assert data["scores"]["exact"]["passed"] == 1
        assert data["scores"]["exact"]["total"] == 2
        assert data["scores"]["exact"]["score"] == 50.0

    def test_patch_merge(self, tmp_path):
        run_json = tmp_path / "run.json"
        run_json.write_text(json.dumps({
            "run_id": "test",
            "scores": {
                "exact": {"grader": "exact", "score": 50.0, "passed": 1, "total": 2,
                          "passed_task_ids": ["t1"], "failed_task_ids": ["t2"]},
            },
        }))

        multi = {
            "contains": [
                GradeResult("t1", True, 1.0, "42", "42", "contains"),
                GradeResult("t2", True, 1.0, "99", "ans: 99", "contains"),
            ],
        }
        data = patch_run_json_scores(run_json, multi, merge=True)
        # Both graders present
        assert "exact" in data["scores"]
        assert "contains" in data["scores"]
        assert data["scores"]["contains"]["passed"] == 2

    def test_legacy_fields_from_first_grader(self, tmp_path):
        run_json = tmp_path / "run.json"
        run_json.write_text(json.dumps({"run_id": "test", "scores": {}}))

        multi = {
            "exact": [GradeResult("t1", True, 1.0, "42", "42", "exact")],
        }
        data = patch_run_json_scores(run_json, multi)
        assert data["score"] == 100.0
        assert data["passed"] == 1
        assert data["total_graded"] == 1


# ============================================================================
# Legacy helper backward compat
# ============================================================================

class TestLegacyCompat:

    def test_grade_with_pipeline(self):
        passed, method = grade_with_pipeline("42", "42")
        assert passed is True
        assert method == "exact"

    def test_get_grader_preset(self):
        preset = get_grader_preset("strict")
        assert len(preset) == 2

    def test_match_functions_still_work(self):
        assert exact_match("a", "a") is True
        assert normalized_match("A B", "a  b") is True
        assert numeric_match("2.15", "2.1498") is True
        assert contains_match("answer is 42", "42") is True
        assert fuzzy_match("hello world", "hello  world") is True

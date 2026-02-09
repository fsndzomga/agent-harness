"""Grading orchestration — shared by ``run``, ``continue``, and ``grade`` commands.

This module contains the logic for:
  - Running one or more graders over a set of submissions
  - Saving / loading / merging ``grades.json`` (multi-grader format)
  - Updating ``scores`` in an existing ``run.json``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..benchmarks.base import GradeResult
from ..benchmarks.graders import Grader, resolve_graders


# ============================================================================
# Core: run graders and collect results
# ============================================================================

def run_graders(
    graders: list[Grader],
    submissions: list[dict[str, Any]],
    failed_tasks: list[dict[str, Any]] | None = None,
) -> dict[str, list[GradeResult]]:
    """Run every grader in *graders* over *submissions*.

    Args:
        graders: Grader instances (e.g. from ``resolve_graders``).
        submissions: List of dicts with keys ``task_id``, ``submission``,
            ``expected``, and optionally ``question``.
        failed_tasks: Tasks that had no submission (status != success).
            Each dict must have ``task_id`` and ``expected``.

    Returns:
        ``{grader.name: [GradeResult, ...]}`` — one entry per grader.
    """
    failed_tasks = failed_tasks or []
    results: dict[str, list[GradeResult]] = {}

    for grader in graders:
        grade_list: list[GradeResult] = []

        # Grade real submissions
        for sub in submissions:
            extra = {
                k: v
                for k, v in sub.items()
                if k not in ("task_id", "submission", "expected")
            }
            gr = grader.grade(
                task_id=sub["task_id"],
                submission=sub["submission"],
                expected=sub["expected"],
                **extra,
            )
            grade_list.append(gr)

        # Add failed tasks (no submission) as incorrect
        for ft in failed_tasks:
            grade_list.append(GradeResult(
                task_id=ft["task_id"],
                passed=False,
                score=0.0,
                expected=ft.get("expected", ""),
                actual=ft.get("error", "no submission"),
                method="failed",
            ))

        results[grader.name] = grade_list

    return results


# ============================================================================
# grades.json I/O (multi-grader format)
# ============================================================================

def _grade_to_dict(g: GradeResult) -> dict:
    d = {
        "task_id": g.task_id,
        "passed": g.passed,
        "score": g.score,
        "expected": g.expected,
        "actual": g.actual,
        "method": g.method,
    }
    if g.details:
        d["details"] = g.details
    return d


def save_grades(
    grades_path: Path,
    multi_grades: dict[str, list[GradeResult]],
    *,
    merge: bool = False,
) -> None:
    """Save grades to *grades_path* in multi-grader format.

    If *merge* is True and the file already exists, new grader entries are
    **added** alongside existing ones (existing grader keys are overwritten).
    """
    # Multi-grader format: {"grader_name": [grade_dicts], ...}
    existing: dict[str, list[dict]] = {}
    if merge and grades_path.exists():
        existing = load_grades_raw(grades_path)

    for grader_name, grade_list in multi_grades.items():
        existing[grader_name] = [_grade_to_dict(g) for g in grade_list]

    grades_path.write_text(json.dumps(existing, indent=2))


def load_grades_raw(grades_path: Path) -> dict[str, list[dict]]:
    """Load grades.json, handling both old (list) and new (dict) formats.

    Returns ``{grader_name: [grade_dicts]}``.
    """
    data = json.loads(grades_path.read_text())

    if isinstance(data, list):
        # Legacy format — single flat list
        return {"default": data}

    if isinstance(data, dict):
        return data

    return {}


def load_grades(grades_path: Path) -> dict[str, list[GradeResult]]:
    """Load grades.json into ``{grader_name: [GradeResult]}``."""
    raw = load_grades_raw(grades_path)
    result: dict[str, list[GradeResult]] = {}
    for grader_name, glist in raw.items():
        result[grader_name] = [
            GradeResult(
                task_id=g["task_id"],
                passed=g["passed"],
                score=g.get("score", 1.0 if g["passed"] else 0.0),
                expected=g.get("expected", ""),
                actual=g.get("actual", ""),
                method=g.get("method", "unknown"),
                details=g.get("details"),
            )
            for g in glist
        ]
    return result


# ============================================================================
# run.json scores patching
# ============================================================================

def patch_run_json_scores(
    run_json_path: Path,
    multi_grades: dict[str, list[GradeResult]],
    *,
    merge: bool = False,
) -> dict:
    """Update the ``scores`` dict inside an existing ``run.json``.

    If *merge* is True, new grader entries are added alongside existing ones.
    Returns the updated run dict (also written to disk).
    """
    run_data = json.loads(run_json_path.read_text())

    existing_scores: dict = run_data.get("scores", {})
    if not merge:
        existing_scores = {}

    for grader_name, glist in multi_grades.items():
        g_passed_ids = [g.task_id for g in glist if g.passed]
        g_failed_ids = [g.task_id for g in glist if not g.passed]
        g_passed = len(g_passed_ids)
        g_total = len(glist)
        g_score = (100 * g_passed / g_total) if g_total > 0 else 0.0

        existing_scores[grader_name] = {
            "grader": grader_name,
            "score": g_score,
            "passed": g_passed,
            "total": g_total,
            "passed_task_ids": g_passed_ids,
            "failed_task_ids": g_failed_ids,
        }

    run_data["scores"] = existing_scores

    # Update legacy fields from first grader
    first_key = next(iter(existing_scores), None)
    if first_key:
        first = existing_scores[first_key]
        run_data["score"] = first["score"]
        run_data["passed"] = first["passed"]
        run_data["total_graded"] = first["total"]
        run_data["passed_task_ids"] = first["passed_task_ids"]
        run_data["failed_task_ids"] = first["failed_task_ids"]

    run_json_path.write_text(json.dumps(run_data, indent=2))
    return run_data


# ============================================================================
# Helpers for building submission / failed-task lists
# ============================================================================

def build_submissions_from_results(
    results: list,
    bench: Any,
    task_data_map: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """Build submissions + failed-task lists from TaskResult objects.

    Returns (submissions, failed_tasks).
    """
    submissions: list[dict] = []
    failed_tasks: list[dict] = []
    submitted_ids: set[str] = set()

    for r in results:
        # r may be a TaskResult or a dict
        task_id = r.task_id if hasattr(r, "task_id") else r["task_id"]
        submission = r.submission if hasattr(r, "submission") else r.get("submission")
        status = r.status if hasattr(r, "status") else r.get("status")
        error = getattr(r, "error", None) or (r.get("error") if isinstance(r, dict) else None)

        expected = bench._answers.get(task_id, "")
        question = task_data_map.get(task_id, {}).get("question", "")

        if submission is not None:
            submissions.append({
                "task_id": task_id,
                "submission": submission,
                "expected": expected,
                "question": question,
            })
            submitted_ids.add(task_id)
        else:
            failed_tasks.append({
                "task_id": task_id,
                "expected": expected,
                "error": error or "no submission",
            })

    return submissions, failed_tasks


def build_submissions_from_merged(
    merged_results_data: list[dict],
    bench: Any,
    task_data_map: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """Build from merged dict-based results (used by ``continue``)."""
    submissions: list[dict] = []
    failed_tasks: list[dict] = []

    for r in merged_results_data:
        task_id = r["task_id"]
        submission = r.get("submission")
        expected = bench._answers.get(task_id, "")
        question = task_data_map.get(task_id, {}).get("question", "")

        if submission is not None:
            submissions.append({
                "task_id": task_id,
                "submission": submission,
                "expected": expected,
                "question": question,
            })
        else:
            failed_tasks.append({
                "task_id": task_id,
                "expected": expected,
                "error": r.get("error", "no submission"),
            })

    return submissions, failed_tasks

"""Grading infrastructure for the v2 architecture.

Provides the ``GradingPipeline`` class (v2) and re-exports all functions
from the original ``grading`` module for backward compatibility.
"""

# v2 API
from .pipeline import GradingPipeline

# Backward-compatible re-exports from the original grading module
from .core import (
    run_graders,
    save_grades,
    load_grades,
    load_grades_raw,
    patch_run_json_scores,
    build_submissions_from_results,
    build_submissions_from_merged,
)

__all__ = [
    # v2
    "GradingPipeline",
    # Legacy
    "run_graders",
    "save_grades",
    "load_grades",
    "load_grades_raw",
    "patch_run_json_scores",
    "build_submissions_from_results",
    "build_submissions_from_merged",
]

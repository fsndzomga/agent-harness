"""Benchmark plugin system."""

from .base import Benchmark, GradeResult
from .registry import get_benchmark, list_benchmarks, get_benchmark_info

__all__ = [
    "Benchmark",
    "GradeResult",
    "get_benchmark",
    "list_benchmarks",
    "get_benchmark_info",
]

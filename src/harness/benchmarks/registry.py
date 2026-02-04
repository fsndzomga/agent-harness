"""Benchmark registry - load benchmarks by name."""

from typing import Any

from .base import Benchmark

# Registry of available benchmarks
_BENCHMARKS: dict[str, type[Benchmark]] = {}


def register_benchmark(cls: type[Benchmark]) -> type[Benchmark]:
    """Decorator to register a benchmark class."""
    _BENCHMARKS[cls.name] = cls
    return cls


def get_benchmark(name: str, **kwargs: Any) -> Benchmark:
    """
    Load a benchmark by name.
    
    Args:
        name: Benchmark name (e.g., "arithmetic", "gaia", "gaia-level1")
        **kwargs: Arguments passed to benchmark constructor
    
    Returns:
        Benchmark instance
    """
    _ensure_registered()
    
    if name not in _BENCHMARKS:
        available = ", ".join(sorted(_BENCHMARKS.keys()))
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    
    return _BENCHMARKS[name](**kwargs)


def list_benchmarks() -> list[str]:
    """Return list of available benchmark names."""
    _ensure_registered()
    return sorted(_BENCHMARKS.keys())


def get_benchmark_info(name: str) -> dict[str, Any]:
    """Get info about a benchmark without instantiating it."""
    _ensure_registered()
    
    if name not in _BENCHMARKS:
        return {}
    
    cls = _BENCHMARKS[name]
    return {
        "name": cls.name,
        "description": getattr(cls, "description", ""),
    }


def _ensure_registered():
    """Ensure all built-in benchmarks are registered."""
    if _BENCHMARKS:
        return
    
    # Import and register built-in benchmarks
    from .arithmetic import ArithmeticBenchmark
    _BENCHMARKS["arithmetic"] = ArithmeticBenchmark
    
    # GAIA benchmarks (may fail if datasets not installed)
    try:
        from .gaia import (
            GAIABenchmark,
            GAIALevel1Benchmark,
            GAIALevel2Benchmark,
            GAIALevel3Benchmark,
        )
        _BENCHMARKS["gaia"] = GAIABenchmark
        _BENCHMARKS["gaia-level1"] = GAIALevel1Benchmark
        _BENCHMARKS["gaia-level2"] = GAIALevel2Benchmark
        _BENCHMARKS["gaia-level3"] = GAIALevel3Benchmark
    except ImportError:
        pass  # datasets not installed, GAIA not available

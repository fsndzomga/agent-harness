"""Run metadata and statistics aggregation.

This module provides a protocol for extracting and aggregating
statistics from trace logs, regardless of provider.
"""

import json
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def get_git_info() -> dict:
    """Get git repository info for reproducibility."""
    info = {
        "commit": None,
        "branch": None,
        "dirty": False,
    }
    
    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()[:12]
        
        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
        
        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["dirty"] = bool(result.stdout.strip())
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # Git not available or timeout
    
    return info


@dataclass
class UsageStats:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Extended token tracking
    cached_tokens: int = 0  # OpenAI/Anthropic cached input tokens (cache reads)
    cache_creation_tokens: int = 0  # Anthropic cache creation (cache writes)
    reasoning_tokens: int = 0  # OpenAI o1/o3, DeepSeek R1 reasoning tokens
    audio_tokens: int = 0  # OpenAI audio input/output
    
    # Provider-specific details (stored as dict for flexibility)
    details: dict = field(default_factory=dict)
    
    def __add__(self, other: "UsageStats") -> "UsageStats":
        # Merge details dicts
        merged_details = {**self.details}
        for k, v in other.details.items():
            if k in merged_details and isinstance(v, (int, float)):
                merged_details[k] = merged_details[k] + v
            else:
                merged_details[k] = v
        
        return UsageStats(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            audio_tokens=self.audio_tokens + other.audio_tokens,
            details=merged_details,
        )


@dataclass
class ModelStats:
    """Per-model statistics."""
    model: str
    requests: int = 0
    usage: UsageStats = field(default_factory=UsageStats)
    cost_usd: float = 0.0
    latency_ms_total: float = 0.0
    latency_ms_avg: float = 0.0
    errors: int = 0


@dataclass 
class TaskStats:
    """Per-task statistics."""
    task_id: str
    status: str  # success, error, timeout
    attempts: int = 1
    usage: UsageStats = field(default_factory=UsageStats)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class RunMetadata:
    """Complete run metadata for database storage."""
    # Identity
    run_id: str
    timestamp: str
    
    # Configuration
    agent: str
    benchmark: str | None
    model: str | None
    grader: str
    
    # Settings
    parallel: int
    max_retries: int
    task_timeout: int
    num_tasks_requested: int | None
    
    # Command used to run (for reproducibility)
    run_command: str | None = None
    
    # Git info for reproducibility
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool = False
    
    # Results
    num_tasks_run: int = 0
    num_tasks_success: int = 0
    num_tasks_failed: int = 0
    
    # Task lists
    successful_task_ids: list[str] = field(default_factory=list)
    failed_task_ids: list[str] = field(default_factory=list)
    
    # Grading
    score: float = 0.0
    passed: int = 0
    total_graded: int = 0
    
    # Aggregated stats
    total_usage: UsageStats = field(default_factory=UsageStats)
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    
    # Per-model breakdown
    model_stats: dict[str, ModelStats] = field(default_factory=dict)
    
    # Per-task stats
    task_stats: list[TaskStats] = field(default_factory=list)
    
    # Duration
    duration_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Convert nested dataclasses
        d["total_usage"] = self._usage_to_dict(self.total_usage)
        d["model_stats"] = {
            k: {**asdict(v), "usage": self._usage_to_dict(v.usage)}
            for k, v in self.model_stats.items()
        }
        d["task_stats"] = [
            {**asdict(t), "usage": self._usage_to_dict(t.usage)}
            for t in self.task_stats
        ]
        return d
    
    @staticmethod
    def _usage_to_dict(usage: UsageStats) -> dict:
        """Convert UsageStats to dict, omitting zero/empty fields."""
        d = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        # Only include extended fields if non-zero
        if usage.cached_tokens:
            d["cached_tokens"] = usage.cached_tokens
        if usage.cache_creation_tokens:
            d["cache_creation_tokens"] = usage.cache_creation_tokens
        if usage.reasoning_tokens:
            d["reasoning_tokens"] = usage.reasoning_tokens
        if usage.audio_tokens:
            d["audio_tokens"] = usage.audio_tokens
        if usage.details:
            d["details"] = usage.details
        return d


def extract_usage_from_response(response: dict) -> UsageStats:
    """
    Extract token usage from a provider response.
    
    Handles multiple provider formats:
    - OpenAI/LiteLLM: response.usage.{prompt_tokens, completion_tokens, total_tokens}
    - OpenAI with details: completion_tokens_details.{reasoning_tokens, audio_tokens}
    - OpenAI cached: prompt_tokens_details.cached_tokens
    - Anthropic: response.usage.{input_tokens, output_tokens, cache_read_input_tokens}
    - DeepSeek: completion_tokens_details.reasoning_tokens
    - Generic: looks for common token fields
    """
    usage = UsageStats()
    details = {}
    
    # Try standard usage object
    usage_obj = response.get("usage", {})
    if not usage_obj and "response" in response:
        # Nested response (our trace format)
        usage_obj = response.get("response", {}).get("usage", {})
    
    if usage_obj:
        # OpenAI/LiteLLM format
        usage.prompt_tokens = usage_obj.get("prompt_tokens", 0) or 0
        usage.completion_tokens = usage_obj.get("completion_tokens", 0) or 0
        usage.total_tokens = usage_obj.get("total_tokens", 0) or 0
        
        # Anthropic format
        if not usage.prompt_tokens:
            usage.prompt_tokens = usage_obj.get("input_tokens", 0) or 0
        if not usage.completion_tokens:
            usage.completion_tokens = usage_obj.get("output_tokens", 0) or 0
        
        # === Extended token tracking ===
        
        # OpenAI prompt_tokens_details (cached tokens)
        prompt_details = usage_obj.get("prompt_tokens_details", {}) or {}
        if prompt_details:
            usage.cached_tokens = prompt_details.get("cached_tokens", 0) or 0
            # Audio input tokens
            if "audio_tokens" in prompt_details:
                details["audio_input_tokens"] = prompt_details["audio_tokens"] or 0
        
        # OpenAI completion_tokens_details (reasoning, audio)
        completion_details = usage_obj.get("completion_tokens_details", {}) or {}
        if completion_details:
            usage.reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0
            usage.audio_tokens = completion_details.get("audio_tokens", 0) or 0
            # Accepted/rejected prediction tokens
            if "accepted_prediction_tokens" in completion_details:
                details["accepted_prediction_tokens"] = completion_details["accepted_prediction_tokens"] or 0
            if "rejected_prediction_tokens" in completion_details:
                details["rejected_prediction_tokens"] = completion_details["rejected_prediction_tokens"] or 0
        
        # Anthropic cache tokens
        if "cache_read_input_tokens" in usage_obj:
            usage.cached_tokens = usage_obj.get("cache_read_input_tokens", 0) or 0
        if "cache_creation_input_tokens" in usage_obj:
            usage.cache_creation_tokens = usage_obj["cache_creation_input_tokens"] or 0
            details["cache_creation_tokens"] = usage.cache_creation_tokens
        
        # DeepSeek reasoning tokens (may be in different location)
        if not usage.reasoning_tokens and "reasoning_tokens" in usage_obj:
            usage.reasoning_tokens = usage_obj["reasoning_tokens"] or 0
        
        # Store any other fields we find for debugging/future use
        known_fields = {
            "prompt_tokens", "completion_tokens", "total_tokens",
            "input_tokens", "output_tokens",
            "prompt_tokens_details", "completion_tokens_details",
            "cache_read_input_tokens", "cache_creation_input_tokens",
            "reasoning_tokens",
        }
        for k, v in usage_obj.items():
            if k not in known_fields and v:
                details[k] = v
        
        # Calculate total if not provided
        if not usage.total_tokens:
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        
        usage.details = details
    
    return usage


def extract_cost_from_response(response: dict) -> float:
    """
    Extract cost from a provider response.
    
    LiteLLM provides cost in _hidden_params, response_cost, or our custom _cost_usd field.
    """
    cost = 0.0
    
    # Our custom field (added by LiteLLM provider)
    if "_cost_usd" in response:
        cost = response["_cost_usd"] or 0
    
    # LiteLLM format in response object
    if not cost and "response" in response:
        resp = response["response"]
        if "_cost_usd" in resp:
            cost = resp["_cost_usd"] or 0
    
    # LiteLLM _hidden_params format
    if not cost and "_hidden_params" in response:
        cost = response["_hidden_params"].get("response_cost", 0) or 0
    
    # Direct cost field
    if not cost:
        cost = response.get("response_cost", 0) or 0
        
    # Nested in response
    if not cost and "response" in response:
        resp = response["response"]
        if "_hidden_params" in resp:
            cost = resp["_hidden_params"].get("response_cost", 0) or 0
        cost = cost or resp.get("response_cost", 0) or 0
    
    return float(cost)


def extract_model_from_entry(entry: dict) -> str:
    """Extract model name from a trace entry."""
    # Try request first
    model = entry.get("request", {}).get("model", "")
    
    # Try response
    if not model:
        model = entry.get("response", {}).get("model", "")
    
    # Try top-level
    if not model:
        model = entry.get("model", "unknown")
    
    return model


def aggregate_trace_file(trace_path: Path) -> tuple[UsageStats, float, float, str | None, int]:
    """
    Aggregate stats from a single trace file.
    
    Returns: (usage, cost, latency_ms, model, request_count)
    """
    usage = UsageStats()
    cost = 0.0
    latency = 0.0
    model = None
    requests = 0
    
    if not trace_path.exists():
        return usage, cost, latency, model, requests
    
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Only process completion entries
            if entry.get("type") != "completion":
                continue
            
            requests += 1
            
            # Extract model
            if not model:
                model = extract_model_from_entry(entry)
            
            # Extract usage
            entry_usage = extract_usage_from_response(entry)
            usage = usage + entry_usage
            
            # Extract cost
            cost += extract_cost_from_response(entry)
            
            # Extract latency
            latency += entry.get("latency_ms", 0) or 0
    
    return usage, cost, latency, model, requests


def aggregate_run_stats(
    output_dir: Path,
    results: list[Any],  # TaskResult objects
    grade_results: list[Any] | None = None,  # GradeResult objects
    config: dict | None = None,
) -> RunMetadata:
    """
    Aggregate all statistics for a run.
    
    Args:
        output_dir: Directory containing trace files
        results: List of TaskResult objects from the run
        grade_results: Optional list of GradeResult objects
        config: Run configuration dict
    """
    config = config or {}
    
    # Get git info for reproducibility
    git_info = get_git_info()
    
    run = RunMetadata(
        run_id=str(uuid.uuid4())[:8],
        timestamp=datetime.utcnow().isoformat() + "Z",
        agent=config.get("agent", "unknown"),
        benchmark=config.get("benchmark"),
        model=config.get("model"),
        grader=config.get("grader", "default"),
        parallel=config.get("parallel", 1),
        max_retries=config.get("max_retries", 3),
        task_timeout=config.get("task_timeout", 300),
        num_tasks_requested=config.get("num_tasks"),
        run_command=config.get("run_command"),
        git_commit=git_info.get("commit"),
        git_branch=git_info.get("branch"),
        git_dirty=git_info.get("dirty", False),
    )
    
    # Aggregate from results
    model_stats: dict[str, ModelStats] = {}
    
    for result in results:
        task_id = result.task_id
        is_success = result.status == "success"
        
        run.num_tasks_run += 1
        if is_success:
            run.num_tasks_success += 1
            run.successful_task_ids.append(task_id)
        else:
            run.num_tasks_failed += 1
            run.failed_task_ids.append(task_id)
        
        # Find trace file
        trace_path = output_dir / f"trace_{task_id}.jsonl"
        usage, cost, latency, model, requests = aggregate_trace_file(trace_path)
        
        # Task stats
        task_stat = TaskStats(
            task_id=task_id,
            status=result.status,
            attempts=getattr(result, 'attempts', 1),
            usage=usage,
            cost_usd=cost,
            latency_ms=latency,
            error=getattr(result, 'error', None),
        )
        run.task_stats.append(task_stat)
        
        # Aggregate totals
        run.total_usage = run.total_usage + usage
        run.total_cost_usd += cost
        run.total_latency_ms += latency
        
        # Per-model stats
        model_key = model or config.get("model", "unknown")
        if model_key not in model_stats:
            model_stats[model_key] = ModelStats(model=model_key)
        
        ms = model_stats[model_key]
        ms.requests += requests
        ms.usage = ms.usage + usage
        ms.cost_usd += cost
        ms.latency_ms_total += latency
        if not is_success:
            ms.errors += 1
    
    # Calculate averages
    for ms in model_stats.values():
        if ms.requests > 0:
            ms.latency_ms_avg = ms.latency_ms_total / ms.requests
    
    run.model_stats = model_stats
    
    # Aggregate grading results
    if grade_results:
        run.total_graded = len(grade_results)
        run.passed = sum(1 for g in grade_results if g.passed)
        if run.total_graded > 0:
            run.score = 100 * run.passed / run.total_graded
    
    return run


def save_run_metadata(run: RunMetadata, output_dir: Path) -> Path:
    """Save run metadata to run.json."""
    run_path = output_dir / "run.json"
    run_path.write_text(json.dumps(run.to_dict(), indent=2))
    return run_path

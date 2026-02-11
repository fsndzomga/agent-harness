"""Run metadata and statistics aggregation.

This module provides a protocol for extracting and aggregating
statistics from trace logs, regardless of provider.

It includes both the complex RunMetadata for detailed tracking
and the simplified RunRecord for flexible agent-driven metadata.
"""

import json
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
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
class RunRecord:
    """Simple run record for flexible agent metadata tracking.
    
    This is a simplified alternative to RunMetadata that lets agents
    self-report whatever metadata matters to them. The harness tracks
    the essential information (results, costs) while agents populate
    agent_metadata however they want for their own ablation studies.
    """
    run_id: str
    benchmark: str
    task_id: str
    agent_name: str
    agent_metadata: dict[str, Any]  # Agent self-reports whatever it wants
    score: float
    cost_usd: float
    model_costs: dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        """Create from JSON dict."""
        # Handle timestamp conversion
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()
        
        return cls(
            run_id=data.get('run_id', str(uuid.uuid4())),
            benchmark=data.get('benchmark', ''),
            task_id=data.get('task_id', ''),
            agent_name=data.get('agent_name', ''),
            agent_metadata=data.get('agent_metadata', {}),
            score=data.get('score', 0.0),
            cost_usd=data.get('cost_usd', 0.0),
            model_costs=data.get('model_costs', {}),
            timestamp=timestamp
        )


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
    status: str  # completed, errored (execution status, not grading result)
    attempts: int = 1
    usage: UsageStats = field(default_factory=UsageStats)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None
    metrics: dict | None = None  # Agent-reported KPIs for this task
    trace: list[dict] = field(default_factory=list)  # Raw trace entries


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
    
    # Execution results
    num_tasks_run: int = 0
    num_tasks_completed: int = 0  # Tasks that ran without crashing
    num_tasks_errored: int = 0  # Tasks that crashed/timed out
    completed_task_ids: list[str] = field(default_factory=list)  # Tasks that executed successfully
    error_task_ids: list[str] = field(default_factory=list)  # Tasks that crashed/timed out
    
    # Grading results — multi-grader scores
    # ``scores`` is the canonical store: {grader_name: {score, passed, total, ...}}
    scores: dict[str, dict] = field(default_factory=dict)
    
    # Legacy / backward-compat fields (populated from the *first* grader)
    score: float = 0.0
    passed: int = 0
    total_graded: int = 0
    passed_task_ids: list[str] = field(default_factory=list)  # Tasks that passed grading
    failed_task_ids: list[str] = field(default_factory=list)  # Tasks that failed grading
    
    # Aggregated stats (total across all models)
    total_usage: UsageStats = field(default_factory=UsageStats)
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    
    # Per-model usage breakdown (simplified view)
    usage_by_model: dict[str, dict] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    
    # Per-model breakdown (full stats)
    model_stats: dict[str, ModelStats] = field(default_factory=dict)
    
    # Per-task stats
    task_stats: list[TaskStats] = field(default_factory=list)
    
    # Aggregated agent metrics (sum/avg of per-task metrics)
    agent_metrics: dict = field(default_factory=dict)
    
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
    trace_entries = []
    
    if not trace_path.exists():
        return usage, cost, latency, model, requests, trace_entries
    
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Store all trace entries
            trace_entries.append(entry)
            
            # Only process completion entries for stats
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
    
    return usage, cost, latency, model, requests, trace_entries


def aggregate_agent_metrics(metrics_list: list[dict]) -> dict:
    """
    Aggregate agent-reported metrics across all tasks.
    
    Aggregation rules:
    - Numeric values: sum and compute average
    - Lists: concatenate
    - Counters (dict with numeric values): sum per key
    - Other: collect unique values
    
    Returns dict with:
    - {metric}_total: sum of numeric values
    - {metric}_avg: average of numeric values
    - {metric}_count: count of tasks reporting this metric
    - Plus any non-numeric aggregations
    """
    if not metrics_list:
        return {}
    
    agg: dict = {}
    counts: dict[str, int] = {}
    
    for metrics in metrics_list:
        if not metrics:
            continue
            
        for key, value in metrics.items():
            if key not in counts:
                counts[key] = 0
            counts[key] += 1
            
            if isinstance(value, (int, float)):
                # Numeric: track sum
                sum_key = f"{key}_total"
                agg[sum_key] = agg.get(sum_key, 0) + value
                
            elif isinstance(value, list):
                # List: concatenate
                list_key = f"{key}_all"
                if list_key not in agg:
                    agg[list_key] = []
                agg[list_key].extend(value)
                
            elif isinstance(value, dict):
                # Dict: assume it's counters, sum per key
                dict_key = f"{key}_totals"
                if dict_key not in agg:
                    agg[dict_key] = {}
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        agg[dict_key][k] = agg[dict_key].get(k, 0) + v
                    else:
                        # Non-numeric dict value: collect
                        if k not in agg[dict_key]:
                            agg[dict_key][k] = []
                        if isinstance(agg[dict_key][k], list):
                            agg[dict_key][k].append(v)
            else:
                # Other: collect unique values
                set_key = f"{key}_values"
                if set_key not in agg:
                    agg[set_key] = []
                if value not in agg[set_key]:
                    agg[set_key].append(value)
    
    # Compute averages for numeric metrics
    for key, count in counts.items():
        sum_key = f"{key}_total"
        if sum_key in agg:
            agg[f"{key}_avg"] = agg[sum_key] / count
            agg[f"{key}_count"] = count
    
    return agg


def aggregate_run_stats(
    output_dir: Path,
    results: list[Any],  # TaskResult objects
    grade_results: list[Any] | None = None,  # GradeResult objects (legacy single-grader)
    grade_results_multi: dict[str, list[Any]] | None = None,  # {grader_name: [GradeResult]}
    config: dict | None = None,
) -> RunMetadata:
    """
    Aggregate all statistics for a run.
    
    Args:
        output_dir: Directory containing trace files
        results: List of TaskResult objects from the run
        grade_results: Optional list of GradeResult objects (single grader, legacy)
        grade_results_multi: Optional dict of grader_name → [GradeResult] (multi-grader)
        config: Run configuration dict
    """
    config = config or {}
    
    # Get git info for reproducibility
    git_info = get_git_info()
    
    # Use provided run_id or generate one
    run_id = config.get("run_id") or str(uuid.uuid4())[:8]
    
    run = RunMetadata(
        run_id=run_id,
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
        is_completed = result.status == "completed"
        
        run.num_tasks_run += 1
        if is_completed:
            run.num_tasks_completed += 1
            run.completed_task_ids.append(task_id)
        else:
            run.num_tasks_errored += 1
            run.error_task_ids.append(task_id)
        
        # Find trace file and extract stats + raw entries
        trace_path = output_dir / f"trace_{task_id}.jsonl"
        usage, cost, latency, model, requests, trace_entries = aggregate_trace_file(trace_path)
        
        # Get agent-reported metrics for this task
        task_metrics = getattr(result, 'metrics', None)
        
        # Task stats (includes full trace)
        task_stat = TaskStats(
            task_id=task_id,
            status=result.status,
            attempts=getattr(result, 'attempts', 1),
            usage=usage,
            cost_usd=cost,
            latency_ms=latency,
            error=getattr(result, 'error', None),
            metrics=task_metrics,
            trace=trace_entries,
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
        if not is_completed:
            ms.errors += 1
    
    # Calculate averages
    for ms in model_stats.values():
        if ms.requests > 0:
            ms.latency_ms_avg = ms.latency_ms_total / ms.requests
    
    run.model_stats = model_stats
    
    # Build simplified per-model views
    for model_name, ms in model_stats.items():
        run.usage_by_model[model_name] = {
            "prompt_tokens": ms.usage.prompt_tokens,
            "completion_tokens": ms.usage.completion_tokens,
            "total_tokens": ms.usage.total_tokens,
        }
        run.cost_by_model[model_name] = ms.cost_usd
    
    # Aggregate grading results (multi-grader aware)
    #
    # Three paths depending on what the caller passed:
    #   1. grade_results_multi  – new multi-grader dict {name: [GradeResult]}
    #   2. grade_results        – legacy single-grader list [GradeResult]
    #   3. neither              – no grading
    all_grades: dict[str, list] = {}
    if grade_results_multi:
        all_grades = grade_results_multi
    elif grade_results:
        # Wrap legacy single list under the grader name from config
        grader_name = config.get("grader", "default")
        all_grades = {grader_name: grade_results}

    first_grader_done = False
    for grader_name, glist in all_grades.items():
        g_passed_ids: list[str] = []
        g_failed_ids: list[str] = []
        g_passed = 0
        g_total = len(glist)
        for g in glist:
            if g.passed:
                g_passed += 1
                g_passed_ids.append(g.task_id)
            else:
                g_failed_ids.append(g.task_id)
        g_score = (100 * g_passed / g_total) if g_total > 0 else 0.0

        run.scores[grader_name] = {
            "grader": grader_name,
            "score": g_score,
            "passed": g_passed,
            "total": g_total,
            "passed_task_ids": g_passed_ids,
            "failed_task_ids": g_failed_ids,
        }

        # Populate legacy flat fields from the FIRST grader for backward compat
        if not first_grader_done:
            run.score = g_score
            run.passed = g_passed
            run.total_graded = g_total
            run.passed_task_ids = g_passed_ids
            run.failed_task_ids = g_failed_ids
            first_grader_done = True
    
    # Aggregate agent metrics across all tasks
    run.agent_metrics = aggregate_agent_metrics([t.metrics for t in run.task_stats if t.metrics])
    
    return run


def save_run_metadata(run: RunMetadata, output_dir: Path) -> Path:
    """Save run metadata to run.json."""
    run_path = output_dir / "run.json"
    run_path.write_text(json.dumps(run.to_dict(), indent=2))
    return run_path


class RunRecordStore:
    """Simple storage and querying for RunRecord instances.
    
    This provides a lightweight way to store and query run records
    for ablation analysis without imposing structure on agent metadata.
    """
    
    def __init__(self, storage_path: Path | str = None):
        """Initialize store with optional storage path.
        
        Args:
            storage_path: Path to JSONL file for storage. If None, uses in-memory only.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._records: list[RunRecord] = []
        self._load_records()
    
    def _load_records(self):
        """Load existing records from storage."""
        if self.storage_path and self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self._records.append(RunRecord.from_dict(data))
            except (json.JSONDecodeError, FileNotFoundError):
                # If file is corrupted or doesn't exist, start fresh
                self._records = []
    
    def add_record(self, record: RunRecord) -> None:
        """Add a new run record."""
        self._records.append(record)
        self._persist_record(record)
    
    def _persist_record(self, record: RunRecord) -> None:
        """Append record to storage file."""
        if self.storage_path:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.storage_path, 'a') as f:
                f.write(json.dumps(record.to_dict()) + '\n')
    
    def query(self, **filters) -> list[RunRecord]:
        """Query records with simple filtering.
        
        Args:
            **filters: Key-value pairs to filter on. Supports:
                - Direct field matching: benchmark="arithmetic"
                - Agent metadata matching: Use 'agent_metadata__key' syntax
        
        Returns:
            List of matching RunRecord instances
        """
        results = []
        
        for record in self._records:
            matches = True
            
            for key, value in filters.items():
                if key.startswith('agent_metadata__'):
                    # Query on agent metadata
                    meta_key = key[len('agent_metadata__'):]
                    if meta_key not in record.agent_metadata or record.agent_metadata[meta_key] != value:
                        matches = False
                        break
                elif hasattr(record, key):
                    # Query on direct field  
                    if getattr(record, key) != value:
                        matches = False
                        break
                else:
                    # Unknown field, no match
                    matches = False
                    break
            
            if matches:
                results.append(record)
        
        return results
    
    def get_benchmarks(self) -> set[str]:
        """Get all unique benchmarks in the store."""
        return {r.benchmark for r in self._records}
    
    def get_agents(self) -> set[str]:
        """Get all unique agent names in the store."""
        return {r.agent_name for r in self._records}
    
    def get_metadata_keys(self, agent_name: str = None) -> set[str]:
        """Get all unique keys used in agent_metadata.
        
        Args:
            agent_name: If provided, only return keys for this agent
        """
        keys = set()
        for record in self._records:
            if agent_name is None or record.agent_name == agent_name:
                keys.update(record.agent_metadata.keys())
        return keys
    
    def stats(self) -> dict[str, Any]:
        """Get basic statistics about the store."""
        return {
            "total_records": len(self._records),
            "benchmarks": list(self.get_benchmarks()),
            "agents": list(self.get_agents()),
            "date_range": {
                "earliest": min(r.timestamp for r in self._records) if self._records else None,
                "latest": max(r.timestamp for r in self._records) if self._records else None
            }
        }


def create_run_record(
    run_id: str,
    benchmark: str, 
    task_id: str,
    agent_name: str,
    score: float,
    cost_usd: float = 0.0,
    model_costs: dict[str, float] = None,
    agent_metadata: dict[str, Any] = None,
    timestamp: datetime = None
) -> RunRecord:
    """Create a RunRecord with sensible defaults.
    
    Args:
        run_id: Unique run identifier
        benchmark: Name of benchmark
        task_id: Task identifier
        agent_name: Name of the agent
        score: Final score (0-100)
        cost_usd: Total cost in USD
        model_costs: Per-model cost breakdown
        agent_metadata: Agent-specific metadata (planning strategy, tools, etc.)
        timestamp: When the run occurred (defaults to now)
    """
    return RunRecord(
        run_id=run_id,
        benchmark=benchmark,
        task_id=task_id,
        agent_name=agent_name,
        score=score,
        cost_usd=cost_usd,
        model_costs=model_costs or {},
        agent_metadata=agent_metadata or {},
        timestamp=timestamp or datetime.now(timezone.utc)
    )


def run_metadata_to_run_records(metadata: RunMetadata) -> list[RunRecord]:
    """Convert RunMetadata to a list of RunRecord instances (one per task)."""
    records = []
    
    for task_stat in metadata.task_stats:
        # Get score for this task from the first grader
        task_score = 0.0
        if metadata.scores:
            first_grader_results = list(metadata.scores.values())[0]
            # This is a simplification - we'd need to match task_id to grading results
            # For now, just use the overall score
            task_score = first_grader_results.get('score', 0.0)
        else:
            task_score = metadata.score if metadata.total_graded > 0 else 0.0
        
        # Extract agent metadata from task metrics
        agent_metadata = task_stat.metrics or {}
        
        # Add some run-level context that might be useful for ablations
        agent_metadata.update({
            'model': metadata.model,
            'parallel': metadata.parallel,
            'max_retries': metadata.max_retries,
            'task_timeout': metadata.task_timeout,
        })
        
        record = RunRecord(
            run_id=metadata.run_id,
            benchmark=metadata.benchmark or "unknown",
            task_id=task_stat.task_id,
            agent_name=metadata.agent,
            score=task_score,
            cost_usd=task_stat.cost_usd,
            model_costs={metadata.model: task_stat.cost_usd} if metadata.model else {},
            agent_metadata=agent_metadata,
            timestamp=datetime.fromisoformat(metadata.timestamp.replace('Z', '+00:00')) if isinstance(metadata.timestamp, str) else datetime.now(timezone.utc)
        )
        
        records.append(record)
    
    return records

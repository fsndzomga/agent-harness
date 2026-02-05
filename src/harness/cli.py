"""CLI for agent harness."""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import click


def slugify(text: str, max_len: int = 30) -> str:
    """Convert text to filesystem-safe slug."""
    # Take last component if path-like (e.g., openrouter/deepseek/model -> model)
    if "/" in text:
        text = text.split("/")[-1]
    # Remove extension
    text = re.sub(r"\.[^.]+$", "", text)
    # Replace non-alphanumeric with dashes
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text)
    # Remove leading/trailing dashes
    text = text.strip("-").lower()
    # Truncate
    return text[:max_len].rstrip("-")


def generate_run_id(
    benchmark: str | None = None,
    agent: str | None = None,
    model: str | None = None,
) -> str:
    """
    Generate a unique, descriptive run ID.
    
    Format: {benchmark}_{agent}_{model}_{YYYYMMDD_HHMMSS}_{random6}
    Example: gaia-level1_simple-qa-agent_deepseek-chat_20260204_224800_a3f2b1
    
    This provides:
    - Human-readable identification of what was run
    - Chronological sortability via timestamp
    - Collision resistance via timestamp + 24-bit random (< 1 in 16M same-second collision)
    - Billions of runs without practical collision risk
    """
    parts = []
    
    if benchmark:
        parts.append(slugify(benchmark, 20))
    
    if agent:
        parts.append(slugify(Path(agent).stem, 25))
    
    if model:
        parts.append(slugify(model, 25))
    
    # Timestamp for sortability and uniqueness
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts.append(ts)
    
    # Random suffix for same-second uniqueness (6 hex = 24 bits)
    rand_suffix = uuid.uuid4().hex[:6]
    parts.append(rand_suffix)
    
    return "_".join(parts)

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

from .protocol import Task
from .runner import AgentRunner
from .logging import TraceLogger


@click.group()
@click.version_option()
def cli():
    """Agent evaluation harness - run AI agents on benchmarks."""
    pass


@cli.command("run-one")
@click.option("--agent", required=True, type=click.Path(exists=True), help="Path to agent")
@click.option("--task", required=True, help="Task as JSON string")
@click.option("--timeout", default=300, help="Timeout in seconds")
@click.option("--output", "-o", type=click.Path(), help="Output directory for traces")
def run_one(agent: str, task: str, timeout: int, output: str | None):
    """Run agent on a single task."""
    
    task_data = json.loads(task)
    task_obj = Task.from_dict(task_data)
    
    # Set up logging if output specified
    logger = None
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = str(uuid.uuid4())[:8]
        logger = TraceLogger(
            output_path=output_dir / "trace.jsonl",
            run_id=run_id,
        )
    
    runner = AgentRunner(Path(agent), logger=logger)
    
    click.echo(f"Running {agent} on task {task_obj.id}...")
    
    try:
        result = runner.run(task_obj, timeout=timeout)
        click.echo(f"✓ Submission: {result.answer}")
        
        if output:
            # Save result
            result_path = Path(output) / "result.json"
            result_path.write_text(json.dumps({
                "task_id": result.task_id,
                "submission": result.answer,
            }, indent=2))
            click.echo(f"Trace: {output}/trace.jsonl")
            
    except Exception as e:
        click.echo(f"✗ Failed: {e}", err=True)
        raise SystemExit(1)
    finally:
        if logger:
            logger.close()


@cli.command("run")
@click.option("--agent", required=True, type=click.Path(exists=True), help="Path to agent")
@click.option("--benchmark", "-b", help="Benchmark name (e.g., 'arithmetic', 'gaia')")
@click.option("--tasks-file", type=click.Path(exists=True), help="JSONL file with tasks (alternative to --benchmark)")
@click.option("--output", "-o", type=click.Path(), default="./results", help="Base output directory (default: ./results)")
@click.option("--run-id", help="Custom run ID (default: auto-generated)")
@click.option("--parallel", "-p", default=10, help="Max parallel tasks")
@click.option("--max-retries", default=3, help="Max retries for failed tasks")
@click.option("--task-timeout", default=300, help="Timeout per task in seconds")
@click.option("--num-tasks", "-n", type=int, help="Limit number of tasks")
@click.option("--model", "-m", help="Model to use (sets HARNESS_MODEL env var)")
@click.option("--grader", "-g", default="default", 
              type=click.Choice(["exact", "normalized", "numeric", "contains", "fuzzy", "strict", "default", "llm", "llm-fallback"]),
              help="Grader to use: exact, normalized, numeric, contains, fuzzy, strict, default (all), llm, llm-fallback")
@click.option("--grader-model", help="Model for LLM-as-judge grader (defaults to --model)")
def run(
    agent: str,
    benchmark: str | None,
    tasks_file: str | None,
    output: str,
    run_id: str | None,
    parallel: int,
    max_retries: int,
    task_timeout: int,
    num_tasks: int | None,
    model: str | None,
    grader: str,
    grader_model: str | None,
):
    """Run agent on a benchmark or task file."""
    import asyncio
    import os
    
    from .parallel import ParallelRunner, RetryConfig
    
    if not benchmark and not tasks_file:
        raise click.UsageError("Must specify either --benchmark or --tasks-file")
    
    # Determine benchmark name for folder structure
    if benchmark:
        bench_name = benchmark
    elif tasks_file:
        bench_name = Path(tasks_file).stem
    else:
        bench_name = "custom"
    
    # Generate run ID if not provided
    if not run_id:
        run_id = generate_run_id(
            benchmark=bench_name,
            agent=agent,
            model=model,
        )
    
    # Create output directory: {output}/{benchmark}/{run_id}/
    base_output = Path(output)
    output_dir = base_output / bench_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"Run ID: {run_id}")
    click.echo(f"Output: {output_dir}")
    
    # Set model env var if specified
    if model:
        os.environ["HARNESS_MODEL"] = model
    
    # Load tasks
    if tasks_file:
        task_list = []
        with open(tasks_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    task_list.append(Task.from_dict(data))
        bench = None
    else:
        from .benchmarks.registry import get_benchmark
        bench = get_benchmark(benchmark)
        task_list = bench.get_tasks()
        bench_name = benchmark
    
    if num_tasks:
        task_list = task_list[:num_tasks]
    
    click.echo(f"Running {len(task_list)} tasks from {bench_name}")
    click.echo(f"Parallel: {parallel}, Retries: {max_retries}, Timeout: {task_timeout}s")
    
    # Prepare agent env
    agent_env = {}
    if model:
        agent_env["HARNESS_MODEL"] = model
    
    # Run
    runner = ParallelRunner(
        agent_path=Path(agent),
        output_dir=output_dir,
        max_parallel=parallel,
        retry_config=RetryConfig(max_retries=max_retries),
        task_timeout=task_timeout,
        agent_env=agent_env,
    )
    
    async def run_with_progress():
        results = []
        async for result in runner.run_streaming(task_list):
            results.append(result)
        return results
    
    results = asyncio.run(run_with_progress())
    
    # Summary
    success = sum(1 for r in results if r.status == "success")
    failed = len(results) - success
    click.echo(f"\nCompleted: {success}/{len(results)} succeeded, {failed} failed")
    
    # Grade if benchmark has grader
    if bench and hasattr(bench, 'grade'):
        click.echo(f"\nGrading with '{grader}' grader...")
        
        # Custom grading based on --grader option
        from .benchmarks.graders import (
            get_grader_preset, 
            grade_with_pipeline, 
            grade_with_llm_fallback,
            LLMJudge,
        )
        from .benchmarks.base import GradeResult
        
        grade_results = []
        submissions = [
            {"task_id": r.task_id, "submission": r.submission}
            for r in results
            if r.submission is not None
        ]
        
        # Get task data for LLM judge context
        task_data_map = {t.id: t.data for t in task_list}
        
        for sub in submissions:
            task_id = sub["task_id"]
            submission = sub["submission"]
            
            # Get expected answer from benchmark
            expected = bench._answers.get(task_id, "")
            question = task_data_map.get(task_id, {}).get("question", "")
            
            if grader == "llm":
                # Pure LLM-as-judge
                judge = LLMJudge(model=grader_model or model)
                passed, method, _ = judge.grade(submission, expected, question)
            elif grader == "llm-fallback":
                # Deterministic first, then LLM fallback
                passed, method = grade_with_llm_fallback(
                    submission, expected, question, 
                    llm_model=grader_model or model
                )
            else:
                # Deterministic graders
                graders_list = get_grader_preset(grader)
                passed, method = grade_with_pipeline(submission, expected, graders_list)
            
            grade_results.append(GradeResult(
                task_id=task_id,
                passed=passed,
                score=1.0 if passed else 0.0,
                expected=expected,
                actual=submission,
                method=method,
            ))
        
        passed = sum(1 for g in grade_results if g.passed)
        total = len(grade_results)
        score = (100 * passed / total) if total > 0 else 0
        click.echo(f"Score: {passed}/{total} ({score:.1f}%)")
        
        # Save grades
        grades_path = output_dir / "grades.json"
        grades_path.write_text(json.dumps([
            {
                "task_id": g.task_id,
                "passed": g.passed,
                "score": g.score,
                "expected": g.expected,
                "actual": g.actual,
                "method": g.method,
            }
            for g in grade_results
        ], indent=2))
        click.echo(f"Grades: {grades_path}")
    
    # Save summary
    summary = {
        "benchmark": benchmark or tasks_file,
        "model": model,
        "total": len(results),
        "success": success,
        "failed": failed,
        "results": [
            {
                "task_id": r.task_id,
                "status": r.status,
                "submission": r.submission,
                "error": r.error,
                "attempts": r.attempts,
                "duration_ms": r.duration_ms,
            }
            for r in results
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    click.echo(f"Summary: {summary_path}")
    
    # Generate run.json with aggregated metadata
    import sys
    from .run_metadata import aggregate_run_stats, save_run_metadata
    
    run_config = {
        "run_id": run_id,
        "agent": agent,
        "benchmark": benchmark or tasks_file,
        "model": model,
        "grader": grader,
        "parallel": parallel,
        "max_retries": max_retries,
        "task_timeout": task_timeout,
        "num_tasks": num_tasks,
        "run_command": " ".join(sys.argv),
    }
    
    # Get grade results if available
    grade_results_for_run = grade_results if bench and hasattr(bench, 'grade') else None
    
    run_metadata = aggregate_run_stats(
        output_dir=output_dir,
        results=results,
        grade_results=grade_results_for_run,
        config=run_config,
    )
    run_metadata.duration_seconds = sum(r.duration_ms or 0 for r in results) / 1000
    
    run_path = save_run_metadata(run_metadata, output_dir)
    click.echo(f"Run: {run_path}")
    
    # Print usage summary
    usage = run_metadata.total_usage
    if usage.total_tokens > 0:
        usage_parts = [
            f"{usage.prompt_tokens:,} prompt",
            f"{usage.completion_tokens:,} completion",
        ]
        if usage.reasoning_tokens:
            usage_parts.append(f"{usage.reasoning_tokens:,} reasoning")
        if usage.cached_tokens:
            usage_parts.append(f"{usage.cached_tokens:,} cached")
        
        click.echo(f"\nUsage: {usage.total_tokens:,} tokens ({', '.join(usage_parts)})")
    
    if run_metadata.total_cost_usd > 0:
        cost = run_metadata.total_cost_usd
        if cost < 0.0001:
            # Show in millicents for very small costs
            click.echo(f"Cost: ${cost:.6f} ({cost * 100000:.2f}¢ per 1000 tasks)")
        elif cost < 0.01:
            click.echo(f"Cost: ${cost:.4f}")
        else:
            click.echo(f"Cost: ${cost:.2f}")
    
    # Per-model breakdown if multiple models
    if len(run_metadata.model_stats) > 1:
        click.echo("\nPer-model breakdown:")
        for model_name, stats in run_metadata.model_stats.items():
            click.echo(f"  {model_name}: {stats.usage.total_tokens:,} tokens, "
                       f"${stats.cost_usd:.4f}, {stats.requests} requests")


@cli.command("benchmarks")
def list_benchmarks():
    """List available benchmarks."""
    from .benchmarks.registry import list_benchmarks, get_benchmark_info
    
    benchmarks = list_benchmarks()
    click.echo("Available benchmarks:\n")
    
    for name in benchmarks:
        info = get_benchmark_info(name)
        click.echo(f"  {name}")
        if info.get("description"):
            click.echo(f"    {info['description']}")
        if info.get("num_tasks"):
            click.echo(f"    Tasks: {info['num_tasks']}")
        click.echo()


@cli.command("view")
@click.argument("path", type=click.Path(exists=True))
def view(path: str):
    """View results from a run."""
    path = Path(path)
    
    # Load summary
    summary_path = path / "summary.json" if path.is_dir() else path
    if not summary_path.exists():
        click.echo(f"No summary.json found at {path}", err=True)
        raise SystemExit(1)
    
    summary = json.loads(summary_path.read_text())
    
    click.echo(f"Benchmark: {summary.get('benchmark', 'unknown')}")
    click.echo(f"Model: {summary.get('model', 'unknown')}")
    click.echo(f"Total: {summary['total']}, Success: {summary['success']}, Failed: {summary['failed']}")
    
    # Load grades if available
    grades_path = path / "grades.json" if path.is_dir() else path.parent / "grades.json"
    if grades_path.exists():
        grades = json.loads(grades_path.read_text())
        passed = sum(1 for g in grades if g["passed"])
        click.echo(f"Score: {passed}/{len(grades)} ({100*passed/len(grades):.1f}%)")
        
        # Show failures
        failures = [g for g in grades if not g["passed"]]
        if failures:
            click.echo(f"\nFailed tasks ({len(failures)}):")
            for g in failures[:10]:  # Show first 10
                click.echo(f"  {g['task_id']}: expected '{g['expected']}', got '{g['actual']}'")
            if len(failures) > 10:
                click.echo(f"  ... and {len(failures) - 10} more")


def find_run_dir(run_id: str, results_base: Path) -> Path | None:
    """
    Find the run directory for a given run_id.
    
    Searches for:
    1. Exact match: {results_base}/*/{run_id}/run.json
    2. Prefix match: {results_base}/*/*{run_id}*/run.json
    3. Direct path if run_id is a path
    """
    run_id_path = Path(run_id)
    
    # If run_id is a direct path to a directory with run.json
    if run_id_path.is_dir() and (run_id_path / "run.json").exists():
        return run_id_path
    
    # Search under results_base
    if not results_base.is_dir():
        return None
    
    matches = []
    for benchmark_dir in results_base.iterdir():
        if not benchmark_dir.is_dir():
            continue
        for run_dir in benchmark_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if run_dir.name == run_id and (run_dir / "run.json").exists():
                return run_dir  # Exact match
            if run_id in run_dir.name and (run_dir / "run.json").exists():
                matches.append(run_dir)
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Return the most recent (by directory name, which includes timestamp)
        return sorted(matches, key=lambda p: p.name)[-1]
    
    return None


@cli.command("continue")
@click.argument("run_id")
@click.option("--output", "-o", type=click.Path(), default="./results", help="Base results directory (default: ./results)")
@click.option("--parallel", "-p", type=int, help="Override parallel workers")
@click.option("--max-retries", type=int, help="Override max retries")
@click.option("--task-timeout", type=int, help="Override task timeout")
def continue_run(
    run_id: str,
    output: str,
    parallel: int | None,
    max_retries: int | None,
    task_timeout: int | None,
):
    """Continue a failed run by re-running errored tasks.
    
    Reads run.json from a previous run, re-runs tasks that errored during
    execution, and updates all output files (traces, grades, summary, run.json).
    
    RUN_ID can be a run ID, a partial match, or a direct path to a run directory.
    
    \b
    Examples:
        harness continue arithmetic_echo-agent_20260204_225130_5d8519
        harness continue 5d8519
        harness continue ./results/arithmetic/my-run/
    """
    import asyncio
    import os
    import sys

    from .parallel import ParallelRunner, RetryConfig

    results_base = Path(output)

    # Find the run directory
    run_dir = find_run_dir(run_id, results_base)
    if run_dir is None:
        click.echo(f"Could not find run directory for '{run_id}' under {results_base}", err=True)
        raise SystemExit(1)

    # Load run.json
    run_json_path = run_dir / "run.json"
    run_data = json.loads(run_json_path.read_text())
    click.echo(f"Found run: {run_dir}")

    # Get error task IDs (handle both old and new field names)
    error_task_ids = run_data.get("error_task_ids", [])
    if not error_task_ids:
        # Backward compat: old format stored execution failures in failed_task_ids
        # but only if there's no separate error_task_ids field
        if "error_task_ids" not in run_data:
            error_task_ids = run_data.get("failed_task_ids", [])

    if not error_task_ids:
        click.echo("No errored tasks to re-run. Run completed successfully!")
        return

    click.echo(f"Re-running {len(error_task_ids)} errored task(s): {', '.join(error_task_ids)}")

    # Extract config from previous run
    agent = run_data.get("agent")
    benchmark_name = run_data.get("benchmark")
    model = run_data.get("model")
    grader = run_data.get("grader", "default")
    grader_model = run_data.get("grader_model")
    prev_parallel = run_data.get("parallel", 10)
    prev_max_retries = run_data.get("max_retries", 3)
    prev_task_timeout = run_data.get("task_timeout", 300)

    # Allow CLI overrides
    eff_parallel = parallel if parallel is not None else prev_parallel
    eff_max_retries = max_retries if max_retries is not None else prev_max_retries
    eff_task_timeout = task_timeout if task_timeout is not None else prev_task_timeout

    # Validate agent path
    agent_path = Path(agent)
    if not agent_path.exists():
        click.echo(f"Agent not found: {agent}", err=True)
        raise SystemExit(1)

    # Set model env var if specified
    if model:
        os.environ["HARNESS_MODEL"] = model

    # Load tasks from benchmark or tasks-file
    if benchmark_name:
        from .benchmarks.registry import get_benchmark
        bench = get_benchmark(benchmark_name)
        all_tasks = bench.get_tasks()
    else:
        click.echo("Cannot continue: no benchmark specified in original run", err=True)
        raise SystemExit(1)

    # Filter to only the errored tasks
    error_task_set = set(error_task_ids)
    retry_tasks = [t for t in all_tasks if t.id in error_task_set]

    if not retry_tasks:
        click.echo(f"Could not find matching tasks in benchmark '{benchmark_name}' for IDs: {error_task_ids}", err=True)
        raise SystemExit(1)

    found_ids = {t.id for t in retry_tasks}
    missing_ids = error_task_set - found_ids
    if missing_ids:
        click.echo(f"Warning: Could not find tasks for IDs: {', '.join(sorted(missing_ids))}")

    click.echo(f"Parallel: {eff_parallel}, Retries: {eff_max_retries}, Timeout: {eff_task_timeout}s")

    # Delete old trace files for errored tasks (they'll be regenerated)
    for task_id in error_task_ids:
        old_trace = run_dir / f"trace_{task_id}.jsonl"
        if old_trace.exists():
            old_trace.unlink()

    # Prepare agent env
    agent_env = {}
    if model:
        agent_env["HARNESS_MODEL"] = model

    # Run the errored tasks
    runner = ParallelRunner(
        agent_path=agent_path,
        output_dir=run_dir,
        max_parallel=eff_parallel,
        retry_config=RetryConfig(max_retries=eff_max_retries),
        task_timeout=eff_task_timeout,
        agent_env=agent_env,
    )

    async def run_with_progress():
        results = []
        async for result in runner.run_streaming(retry_tasks):
            results.append(result)
        return results

    new_results = asyncio.run(run_with_progress())

    # Report on retry results
    new_success = sum(1 for r in new_results if r.status == "success")
    new_failed = len(new_results) - new_success
    click.echo(f"\nRetry results: {new_success}/{len(new_results)} succeeded, {new_failed} failed")

    # Load existing summary to get old successful results
    old_summary_path = run_dir / "summary.json"
    old_results_data = []
    if old_summary_path.exists():
        old_summary = json.loads(old_summary_path.read_text())
        old_results_data = old_summary.get("results", [])

    # Merge: keep old successful results, replace errored ones with new results
    merged_results_data = []
    for old_r in old_results_data:
        if old_r["task_id"] not in error_task_set:
            merged_results_data.append(old_r)

    # Add new retry results
    for r in new_results:
        merged_results_data.append({
            "task_id": r.task_id,
            "status": r.status,
            "submission": r.submission,
            "error": r.error,
            "attempts": r.attempts,
            "duration_ms": r.duration_ms,
        })

    # Total counts across merged results
    total_success = sum(1 for r in merged_results_data if r["status"] == "success")
    total_failed = len(merged_results_data) - total_success

    # Grade all submissions (not just the retried ones)
    grade_results = None
    if benchmark_name and bench and hasattr(bench, 'grade'):
        click.echo(f"\nGrading with '{grader}' grader...")

        from .benchmarks.graders import (
            get_grader_preset,
            grade_with_pipeline,
            grade_with_llm_fallback,
            LLMJudge,
        )
        from .benchmarks.base import GradeResult

        grade_results = []
        submissions = [
            {"task_id": r["task_id"], "submission": r["submission"]}
            for r in merged_results_data
            if r.get("submission") is not None
        ]

        task_data_map = {t.id: t.data for t in all_tasks}

        for sub in submissions:
            task_id = sub["task_id"]
            submission = sub["submission"]
            expected = bench._answers.get(task_id, "")
            question = task_data_map.get(task_id, {}).get("question", "")

            if grader == "llm":
                judge = LLMJudge(model=grader_model or model)
                passed, method, _ = judge.grade(submission, expected, question)
            elif grader == "llm-fallback":
                passed, method = grade_with_llm_fallback(
                    submission, expected, question,
                    llm_model=grader_model or model,
                )
            else:
                graders_list = get_grader_preset(grader)
                passed, method = grade_with_pipeline(submission, expected, graders_list)

            grade_results.append(GradeResult(
                task_id=task_id,
                passed=passed,
                score=1.0 if passed else 0.0,
                expected=expected,
                actual=submission,
                method=method,
            ))

        g_passed = sum(1 for g in grade_results if g.passed)
        g_total = len(grade_results)
        g_score = (100 * g_passed / g_total) if g_total > 0 else 0
        click.echo(f"Score: {g_passed}/{g_total} ({g_score:.1f}%)")

        # Save grades
        grades_path = run_dir / "grades.json"
        grades_path.write_text(json.dumps([
            {
                "task_id": g.task_id,
                "passed": g.passed,
                "score": g.score,
                "expected": g.expected,
                "actual": g.actual,
                "method": g.method,
            }
            for g in grade_results
        ], indent=2))
        click.echo(f"Grades: {grades_path}")

    # Save updated summary
    summary = {
        "benchmark": benchmark_name,
        "model": model,
        "total": len(merged_results_data),
        "success": total_success,
        "failed": total_failed,
        "results": merged_results_data,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    click.echo(f"Summary: {summary_path}")

    # Rebuild run.json from all trace files
    from .run_metadata import aggregate_run_stats, save_run_metadata
    from .parallel import TaskResult

    # Reconstruct TaskResult objects from merged summary data
    all_task_results = []
    for r in merged_results_data:
        all_task_results.append(TaskResult(
            task_id=r["task_id"],
            status=r["status"],
            submission=r.get("submission"),
            error=r.get("error"),
            attempts=r.get("attempts", 1),
            duration_ms=r.get("duration_ms", 0),
        ))

    run_config = {
        "run_id": run_data["run_id"],
        "agent": agent,
        "benchmark": benchmark_name,
        "model": model,
        "grader": grader,
        "parallel": eff_parallel,
        "max_retries": eff_max_retries,
        "task_timeout": eff_task_timeout,
        "num_tasks": run_data.get("num_tasks_requested"),
        "run_command": f"harness continue {run_id} (original: {run_data.get('run_command', 'N/A')})",
    }

    run_metadata = aggregate_run_stats(
        output_dir=run_dir,
        results=all_task_results,
        grade_results=grade_results,
        config=run_config,
    )
    run_metadata.duration_seconds = sum(r.get("duration_ms", 0) for r in merged_results_data) / 1000

    run_path = save_run_metadata(run_metadata, run_dir)
    click.echo(f"Run: {run_path}")

    # Print usage summary
    usage = run_metadata.total_usage
    if usage.total_tokens > 0:
        usage_parts = [
            f"{usage.prompt_tokens:,} prompt",
            f"{usage.completion_tokens:,} completion",
        ]
        if usage.reasoning_tokens:
            usage_parts.append(f"{usage.reasoning_tokens:,} reasoning")
        if usage.cached_tokens:
            usage_parts.append(f"{usage.cached_tokens:,} cached")
        click.echo(f"\nUsage: {usage.total_tokens:,} tokens ({', '.join(usage_parts)})")

    if run_metadata.total_cost_usd > 0:
        cost = run_metadata.total_cost_usd
        if cost < 0.01:
            click.echo(f"Cost: ${cost:.4f}")
        else:
            click.echo(f"Cost: ${cost:.2f}")

    # Final status
    if new_failed == 0:
        click.echo(f"\nAll {len(error_task_ids)} previously errored tasks now succeeded!")
    else:
        still_errored = [r.task_id for r in new_results if r.status != "success"]
        click.echo(f"\n{new_failed} task(s) still failing: {', '.join(still_errored)}")


DEFAULT_HF_REPO = "fsndzomga/agent-harness-runs"


@cli.command("push")
@click.argument("run_id")
@click.option("--repo", "-r", default=DEFAULT_HF_REPO, show_default=True, help="HuggingFace dataset repo ID")
@click.option("--output", "-o", type=click.Path(), default="./results", help="Base results directory (default: ./results)")
@click.option("--private", is_flag=True, help="Create repo as private if it doesn't exist")
@click.option("--token", envvar="HF_TOKEN", help="HuggingFace API token (default: HF_TOKEN env var)")
def push(run_id: str, repo: str, output: str, private: bool, token: str | None):
    """Push a run to HuggingFace dataset repo.

    Uploads run.json as {run_id}.json to the canonical dataset repo.
    Default repo: fsndzomga/agent-harness-runs

    \b
    Examples:
        harness push my-run-id
        harness push ./results/arithmetic/my-run/
        harness push 5d8519 --private
    """
    from .hf_integration import push_run

    results_base = Path(output)
    run_dir = find_run_dir(run_id, results_base)
    if run_dir is None:
        click.echo(f"Could not find run directory for '{run_id}' under {results_base}", err=True)
        raise SystemExit(1)

    click.echo(f"Pushing {run_dir} to {repo}...")

    try:
        result = push_run(
            run_dir=run_dir,
            repo_id=repo,
            token=token,
            private=private,
        )
        click.echo(f"Pushed run '{result['run_id']}' to {result['repo_url']}")
        click.echo(f"File: {result['file_url']}")
    except ImportError as e:
        click.echo(f"Missing dependency: {e}", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Push failed: {e}", err=True)
        raise SystemExit(1)


@cli.command("pull")
@click.argument("run_id")
@click.option("--repo", "-r", default=DEFAULT_HF_REPO, show_default=True, help="HuggingFace dataset repo ID")
@click.option("--output", "-o", type=click.Path(), default=".", help="Output directory (default: current dir)")
@click.option("--token", envvar="HF_TOKEN", help="HuggingFace API token")
def pull(run_id: str, repo: str, output: str, token: str | None):
    """Pull a run.json from HuggingFace dataset repo.

    \b
    Examples:
        harness pull my-run-id
        harness pull my-run-id -o ./downloads
    """
    from .hf_integration import pull_run

    click.echo(f"Downloading run '{run_id}' from {repo}...")

    try:
        path = pull_run(
            run_id=run_id,
            repo_id=repo,
            output_dir=Path(output),
            token=token,
        )
        click.echo(f"Downloaded: {path}")
    except ImportError as e:
        click.echo(f"Missing dependency: {e}", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Pull failed: {e}", err=True)
        raise SystemExit(1)


@cli.command("runs")
@click.option("--repo", "-r", default=DEFAULT_HF_REPO, show_default=True, help="HuggingFace dataset repo ID")
@click.option("--token", envvar="HF_TOKEN", help="HuggingFace API token")
@click.option("--json-output", "json_out", is_flag=True, help="Output as JSON")
def list_remote_runs(repo: str, token: str | None, json_out: bool):
    """List runs stored in a HuggingFace dataset repo.

    \b
    Examples:
        harness runs
        harness runs --json-output
    """
    from .hf_integration import list_runs

    try:
        runs = list_runs(repo_id=repo, token=token)
    except ImportError as e:
        click.echo(f"Missing dependency: {e}", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Failed to list runs: {e}", err=True)
        raise SystemExit(1)

    if not runs:
        click.echo("No runs found.")
        return

    if json_out:
        click.echo(json.dumps(runs, indent=2))
        return

    # Table display
    click.echo(f"{'Run ID':<45} {'Benchmark':<15} {'Model':<25} {'Score':>6} {'Tasks':>5} {'Cost':>10}")
    click.echo("─" * 110)
    for r in runs:
        run_id = r.get("run_id", "?")[:44]
        bench = (r.get("benchmark") or "?")[:14]
        model = (r.get("model") or "?")[:24]
        score = r.get("score", 0)
        tasks = r.get("num_tasks_run", 0)
        cost = r.get("total_cost_usd", 0)
        click.echo(f"{run_id:<45} {bench:<15} {model:<25} {score:>5.1f}% {tasks:>5} ${cost:>8.4f}")

    click.echo(f"\n{len(runs)} run(s) in {repo}")


if __name__ == "__main__":
    cli()

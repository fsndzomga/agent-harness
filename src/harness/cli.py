"""CLI for agent harness."""

import json
import uuid
from pathlib import Path

import click

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
@click.option("--output", "-o", required=True, type=click.Path(), help="Output directory")
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
    
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        bench_name = tasks_file
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
        click.echo(f"Cost: ${run_metadata.total_cost_usd:.4f}")
    
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


if __name__ == "__main__":
    cli()

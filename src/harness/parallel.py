"""Parallel task runner with retry logic."""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncIterator

from .protocol import Task, Submission
from .runner import AgentRunner
from .logging import TraceLogger


class ErrorType(Enum):
    """Categorize errors to determine retry behavior."""
    RETRYABLE = "retryable"          # 429, 5xx, timeout, overloaded
    NON_RETRYABLE = "non_retryable"  # 400, 401, agent logic error


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1  # ±10% randomization


@dataclass
class TaskResult:
    """Result of running a single task."""
    task_id: str
    status: str  # "success", "failed", "timeout", "cancelled"
    submission: str | None = None
    error: str | None = None
    error_type: ErrorType | None = None
    attempts: int = 1
    duration_ms: int = 0
    trace_file: str | None = None
    metrics: dict | None = None  # Agent-reported KPIs


def classify_error(error: Exception) -> tuple[ErrorType, float | None]:
    """
    Classify an exception to determine if we should retry.
    Returns (error_type, optional_delay_hint).
    """
    error_str = str(error).lower()
    
    # Rate limits - often have delay hints
    if "429" in error_str or "rate" in error_str and "limit" in error_str:
        # Try to extract retry-after from error message
        import re
        match = re.search(r'retry.?after[:\s]+(\d+)', error_str)
        delay_hint = float(match.group(1)) if match else None
        return ErrorType.RETRYABLE, delay_hint
    
    # Server errors
    if any(code in error_str for code in ["500", "502", "503", "504"]):
        return ErrorType.RETRYABLE, None
    
    # Timeouts
    if "timeout" in error_str or "timed out" in error_str:
        return ErrorType.RETRYABLE, None
    
    # Overloaded/capacity
    if "overloaded" in error_str or "capacity" in error_str:
        return ErrorType.RETRYABLE, 5.0  # Wait a bit
    
    # Connection errors (transient)
    if "connection" in error_str and ("reset" in error_str or "refused" in error_str):
        return ErrorType.RETRYABLE, None
    
    # Auth errors - don't retry
    if "401" in error_str or "403" in error_str or "invalid" in error_str and "key" in error_str:
        return ErrorType.NON_RETRYABLE, None
    
    # Bad request - don't retry  
    if "400" in error_str or "invalid" in error_str and "request" in error_str:
        return ErrorType.NON_RETRYABLE, None
    
    # Default: don't retry unknown errors
    return ErrorType.NON_RETRYABLE, None


class ParallelRunner:
    """Runs tasks in parallel with retry logic and progress tracking."""
    
    def __init__(
        self,
        agent_path: Path,
        output_dir: Path,
        max_parallel: int = 10,
        retry_config: RetryConfig | None = None,
        task_timeout: float = 300,
        agent_env: dict[str, str] | None = None,
    ):
        self.agent_path = Path(agent_path)
        self.output_dir = Path(output_dir)
        self.max_parallel = max_parallel
        self.retry_config = retry_config or RetryConfig()
        self.task_timeout = task_timeout
        self.agent_env = agent_env or {}
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Size the thread pool to match parallelism so tasks aren't
        # queued behind others while their asyncio timeout ticks.
        self._executor = ThreadPoolExecutor(max_workers=max_parallel)
        
        # Real-time status file — append-only for crash safety.
        # Each line records one task completion/failure so that
        # interrupted runs can be continued without run.json.
        self._status_path = self.output_dir / "status.jsonl"
        self._status_lock = asyncio.Lock()
        
        # Progress tracking
        self._completed = 0
        self._failed = 0
        self._total = 0
        self._stop_flag = False
    
    async def run_all(self, tasks: list[Task]) -> list[TaskResult]:
        """Run all tasks, return results in order."""
        self._total = len(tasks)
        self._completed = 0  
        self._failed = 0
        self._stop_flag = False
        
        semaphore = asyncio.Semaphore(self.max_parallel)
        
        async def run_with_sem(task: Task) -> TaskResult:
            async with semaphore:
                if self._stop_flag:
                    return TaskResult(
                        task_id=task.id,
                        status="cancelled",
                        error="Run stopped",
                    )
                return await self._run_with_retry(task)
        
        results = await asyncio.gather(*[run_with_sem(t) for t in tasks])
        return results
    
    async def run_streaming(self, tasks: list[Task]) -> AsyncIterator[TaskResult]:
        """Run tasks, yield results as they complete."""
        self._total = len(tasks)
        self._completed = 0
        self._failed = 0
        self._stop_flag = False
        
        semaphore = asyncio.Semaphore(self.max_parallel)
        queue: asyncio.Queue[TaskResult] = asyncio.Queue()
        
        async def run_and_queue(task: Task):
            async with semaphore:
                if self._stop_flag:
                    result = TaskResult(
                        task_id=task.id,
                        status="cancelled",
                        error="Run stopped",
                    )
                else:
                    result = await self._run_with_retry(task)
                await queue.put(result)
        
        # Start all tasks
        tasks_async = [asyncio.create_task(run_and_queue(t)) for t in tasks]
        
        # Yield results as they complete
        for _ in range(len(tasks)):
            result = await queue.get()
            yield result
        
        # Wait for cleanup
        await asyncio.gather(*tasks_async, return_exceptions=True)
    
    async def _run_with_retry(self, task: Task) -> TaskResult:
        """Run a single task with retries."""
        cfg = self.retry_config
        last_error = None
        last_error_type = None
        total_start = time.monotonic()
        
        for attempt in range(1, cfg.max_retries + 1):
            try:
                submission = await self._run_task_async(task)
                duration_ms = int((time.monotonic() - total_start) * 1000)
                
                self._completed += 1
                self._print_progress(task.id, "✓", attempt)
                
                result = TaskResult(
                    task_id=task.id,
                    status="success",
                    submission=submission.answer,
                    attempts=attempt,
                    duration_ms=duration_ms,
                    trace_file=str(self.output_dir / f"trace_{task.id}.jsonl"),
                    metrics=submission.metrics if submission.metrics else None,
                )
                self._write_status(result)
                return result
                
            except Exception as e:
                last_error = str(e)
                error_type, delay_hint = classify_error(e)
                last_error_type = error_type
                
                # Non-retryable errors fail immediately
                if error_type == ErrorType.NON_RETRYABLE:
                    self._failed += 1
                    self._print_progress(task.id, "✗", attempt, last_error)
                    
                    result = TaskResult(
                        task_id=task.id,
                        status="failed",
                        error=last_error,
                        error_type=error_type,
                        attempts=attempt,
                        duration_ms=int((time.monotonic() - total_start) * 1000),
                    )
                    self._write_status(result)
                    return result
                
                # Calculate backoff delay
                if delay_hint is not None:
                    delay = delay_hint
                else:
                    delay = min(
                        cfg.initial_delay * (cfg.exponential_base ** (attempt - 1)),
                        cfg.max_delay,
                    )
                    # Add jitter
                    import random
                    jitter_range = delay * cfg.jitter
                    delay += random.uniform(-jitter_range, jitter_range)
                
                if attempt < cfg.max_retries:
                    self._print_progress(task.id, "↻", attempt, f"{last_error} (retry in {delay:.1f}s)")
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        self._failed += 1
        self._print_progress(task.id, "✗", cfg.max_retries, last_error)
        
        result = TaskResult(
            task_id=task.id,
            status="failed",
            error=f"Failed after {cfg.max_retries} attempts: {last_error}",
            error_type=last_error_type,
            attempts=cfg.max_retries,
            duration_ms=int((time.monotonic() - total_start) * 1000),
        )
        self._write_status(result)
        return result
    
    async def _run_task_async(self, task: Task) -> Submission:
        """Run task in executor (subprocess is blocking)."""
        loop = asyncio.get_event_loop()
        
        # Create per-task logger
        trace_path = self.output_dir / f"trace_{task.id}.jsonl"
        logger = TraceLogger(output_path=trace_path, run_id=task.id)
        
        runner = AgentRunner(self.agent_path, logger=logger, env=self.agent_env)
        
        try:
            # Use our properly-sized executor so every task gets a
            # thread immediately and the subprocess timeout fires
            # before the asyncio safety timeout.
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor, runner.run, task, self.task_timeout
                ),
                timeout=self.task_timeout + 30,  # Safety buffer for process cleanup
            )
            return result
        except asyncio.TimeoutError:
            # The asyncio safety timeout fired (subprocess timeout
            # should have triggered first).  Log it so traces aren't
            # empty and re-raise as a regular TimeoutError.
            logger.log(
                "task_timeout",
                task_id=task.id,
                timeout=self.task_timeout,
                note="asyncio safety timeout — subprocess may still be running",
            )
            raise TimeoutError(
                f"Agent timed out after {self.task_timeout}s (asyncio safety)"
            )
        finally:
            logger.close()
    
    def _write_status(self, result: TaskResult) -> None:
        """Append a task result to status.jsonl (crash-safe, append-only)."""
        entry = {
            "task_id": result.task_id,
            "status": result.status,
            "timestamp": time.time(),
        }
        if result.submission is not None:
            entry["submission"] = result.submission
        if result.error:
            entry["error"] = result.error
        if result.attempts:
            entry["attempts"] = result.attempts
        if result.duration_ms:
            entry["duration_ms"] = result.duration_ms
        if result.metrics:
            entry["metrics"] = result.metrics

        line = json.dumps(entry, separators=(",", ":")) + "\n"
        with open(self._status_path, "a") as f:
            f.write(line)
            f.flush()

    def _print_progress(self, task_id: str, symbol: str, attempt: int, msg: str = ""):
        """Print progress to console."""
        done = self._completed + self._failed
        extra = f" - {msg}" if msg else ""
        # Truncate long messages
        if len(extra) > 80:
            extra = extra[:77] + "..."
        print(f"[{done}/{self._total}] {symbol} {task_id} (attempt {attempt}){extra}")

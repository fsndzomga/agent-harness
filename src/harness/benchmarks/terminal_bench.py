"""
Terminal-Bench Benchmark — container-graded terminal tasks.

Terminal-Bench evaluates agents on real-world terminal/DevOps tasks inside
Docker containers.  Each task provides:
  - An *instruction* (natural-language description of what to do)
  - A Docker environment (``docker-compose.yaml``)
  - A test suite (``run-tests.sh`` + ``tests/``) that checks the final state

Unlike GAIA (where grading compares a string answer), Terminal-Bench grades
by running the test suite **inside the container** after the agent finishes.

This plugin adapts Terminal-Bench to the agent-harness ``Benchmark`` protocol:
  - ``get_tasks()`` enumerates tasks from a TB dataset (registry or local path).
  - ``grade()`` is the standard entry-point but for TB tasks the *real* grading
    happens via ``grade_container()`` which spins up tests inside Docker.
  - A ``_answers`` dict is provided (mapping task_id → ``"PASS"``) so the
    existing grading pipeline has something to compare against; the string
    submission from the agent is informational only.

Datasets
--------
Terminal-Bench ships curated datasets accessible by name + version via the
``terminal-bench`` package's ``Dataset`` class, or from a local directory.

Docker requirement
------------------
A running Docker daemon is required.  The benchmark checks for Docker at
construction time and raises a clear error if it's unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..protocol import Task
from .base import Benchmark, ExecutionMode, ExecutionContext, GradeResult

logger = logging.getLogger(__name__)


class TerminalBenchBenchmark(Benchmark):
    """Terminal-Bench: container-graded terminal tasks.

    Args:
        dataset_name: Name of the TB dataset in the registry (e.g. ``"core"``).
        dataset_version: Version tag (e.g. ``"v0.1.0"``).  Required when
            *dataset_name* is set.
        dataset_path: Explicit local path to a dataset directory.  If set,
            *dataset_name* / *dataset_version* are ignored.
        task_ids: Optional list of task IDs to include (subset).
        n_tasks: Optional cap on the number of tasks.
        difficulty: Filter by difficulty (``"easy"``, ``"medium"``, ``"hard"``).
    """

    name = "terminal-bench"
    description = (
        "Terminal-Bench — real-world terminal tasks graded by "
        "running tests inside Docker containers"
    )
    execution_mode = ExecutionMode.INTERACTIVE

    def __init__(
        self,
        dataset_name: str | None = None,
        dataset_version: str | None = None,
        dataset_path: str | Path | None = None,
        task_ids: list[str] | None = None,
        n_tasks: int | None = None,
        difficulty: str | None = None,
    ):
        if dataset_path is None and dataset_name is None:
            dataset_name = "terminal-bench-core"
            dataset_version = dataset_version or "0.1.1"

        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self._filter_task_ids = task_ids
        self._n_tasks = n_tasks
        self._difficulty = difficulty

        self._tasks: list[Task] | None = None
        self._answers: dict[str, str] = {}
        self._task_dirs: dict[str, Path] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._dataset_obj: Any = None

    # ------------------------------------------------------------------
    # Benchmark protocol
    # ------------------------------------------------------------------

    def get_tasks(self) -> list[Task]:
        if self._tasks is not None:
            return self._tasks

        self._load_dataset()
        return self._tasks  # type: ignore[return-value]

    def grade(self, task: Task, result: Any, context: ExecutionContext) -> GradeResult:
        """String-comparison grading (fallback).

        For Terminal-Bench the *real* grading is container-based via
        ``grade_container()``.  This method provides basic compatibility
        with the harness grading pipeline by treating any non-empty
        submission that contains "PASS" or comes from ``grade_container``
        as passing.
        """
        task_id = task.id
        submission = str(result) if result is not None else ""
        expected = self._answers.get(task_id, "PASS")

        # If submission was set by grade_container, it will be "PASS"/"FAIL"
        passed = submission.strip().upper() == "PASS"

        return GradeResult(
            task_id=task_id,
            passed=passed,
            score=1.0 if passed else 0.0,
            expected=expected,
            actual=submission,
            method="container" if passed else "none",
            details=self._metadata.get(task_id),
        )

    def grade_container(
        self,
        task_id: str,
        container_manager: Any,
        *,
        timeout: float = 120.0,
    ) -> GradeResult:
        """Run tests inside the Docker container and grade.

        This is the primary grading path for Terminal-Bench tasks.  The
        ``ContainerManager.run_tests()`` method copies the test files into
        the container, executes ``run-tests.sh``, and parses the output.

        Args:
            task_id: The task to grade.
            container_manager: A ``ContainerManager`` instance that has a
                running container for this task.
            timeout: Max time (seconds) for test execution.

        Returns:
            A ``GradeResult`` with full test details.
        """
        passed, details = container_manager.run_tests(
            task_id, timeout=timeout,
        )

        return GradeResult(
            task_id=task_id,
            passed=passed,
            score=1.0 if passed else 0.0,
            expected="PASS (all tests)",
            actual="PASS" if passed else "FAIL",
            method="container",
            details={
                **(self._metadata.get(task_id, {})),
                **details,
            },
        )

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self) -> None:
        """Populate ``_tasks``, ``_answers``, ``_task_dirs``, ``_metadata``."""
        self._ensure_terminal_bench()

        from terminal_bench.dataset.dataset import Dataset
        from terminal_bench.handlers.trial_handler import Task as TBTask, TaskPaths

        # Build the Dataset object
        # When a difficulty filter is active, load ALL tasks from the
        # dataset and apply n_tasks *after* filtering — otherwise the
        # Dataset's own n_tasks cap would slice before we filter.
        if self.dataset_path is not None:
            ds = Dataset(path=self.dataset_path)
        else:
            ds_n_tasks = None if self._difficulty else self._n_tasks
            ds = Dataset(
                name=self.dataset_name,
                version=self.dataset_version,
                task_ids=self._filter_task_ids,
                n_tasks=ds_n_tasks,
            )
        self._dataset_obj = ds

        # ds.tasks is a list[Path] — each Path is a task directory
        task_dirs: list[Path] = ds.tasks
        self._tasks = []

        for task_dir in task_dirs:
            task_paths = TaskPaths(task_dir)
            task_id = task_dir.name  # directory name = task ID

            # Load task config
            try:
                tb_task = TBTask.from_yaml(task_paths.task_config_path)
            except Exception as exc:
                logger.warning("Skipping %s: %s", task_id, exc)
                continue

            # Apply difficulty filter
            if self._difficulty is not None:
                if tb_task.difficulty.value.lower() != self._difficulty.lower():
                    continue

            # Apply explicit task_ids filter (Dataset may already do this,
            # but double-check for local paths)
            if (
                self._filter_task_ids is not None
                and task_id not in self._filter_task_ids
            ):
                continue

            task_data: dict[str, Any] = {
                "instruction": tb_task.instruction,
                "task_type": "terminal",
                "task_dir": str(task_dir),
                "difficulty": tb_task.difficulty.value,
                "category": tb_task.category,
                "max_agent_timeout_sec": tb_task.max_agent_timeout_sec,
            }
            if tb_task.tags:
                task_data["tags"] = tb_task.tags

            self._answers[task_id] = "PASS"
            self._task_dirs[task_id] = task_dir
            self._metadata[task_id] = {
                "difficulty": tb_task.difficulty.value,
                "category": tb_task.category,
                "tags": tb_task.tags,
                "parser": tb_task.parser_name.value,
            }

            self._tasks.append(Task(id=task_id, data=task_data))

        # Apply n_tasks cap (for local-path mode where Dataset didn't)
        if self._n_tasks and len(self._tasks) > self._n_tasks:
            self._tasks = self._tasks[: self._n_tasks]

        logger.info(
            "Loaded %d Terminal-Bench tasks (dataset=%s)",
            len(self._tasks),
            self.dataset_name or self.dataset_path,
        )

    def get_task_dir(self, task_id: str) -> Path | None:
        """Return the local directory for a task (needed by ContainerManager)."""
        self.get_tasks()  # ensure loaded
        return self._task_dirs.get(task_id)

    def get_num_tasks(self) -> int:
        return len(self.get_tasks())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_terminal_bench() -> None:
        try:
            import terminal_bench  # noqa: F401
        except ImportError:
            raise ImportError(
                "terminal-bench is required for the terminal-bench benchmark. "
                "Install it with:  pip install terminal-bench"
            )

    @staticmethod
    def _latest_version(dataset_name: str) -> str:
        """Return the default version for a known dataset."""
        # Known stable versions (update when new releases ship)
        known = {
            "terminal-bench-core": "0.1.1",
            "terminal-bench-head": "head",
        }
        return known.get(dataset_name, "head")


class TerminalBenchCoreBenchmark(TerminalBenchBenchmark):
    """Terminal-Bench core dataset."""
    name = "terminal-bench-core"
    description = "Terminal-Bench core dataset — curated terminal tasks"

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("dataset_name", "terminal-bench-core")
        kwargs.setdefault("dataset_version", "0.1.1")
        super().__init__(**kwargs)

"""Base benchmark class and grade result."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum

from ..protocol import Task


class ExecutionMode(Enum):
    """Execution patterns for benchmarks.

    Every benchmark maps to exactly one of these six archetypes.
    """
    DIRECT = "direct"                   # A: Question → Answer
    CODE_SUBMIT = "code_submit"         # B: Problem → Code → Container Tests
    INTERACTIVE = "interactive"         # C: Agent ↔ Container Shell Loop
    CONVERSATIONAL = "conversational"   # D: Agent ↔ Simulated User
    TOOL_USE = "tool_use"               # E: Agent + Live Tool Servers (NEW)
    GUI_AGENT = "gui_agent"             # F: Agent in VM/Game World (NEW)


@dataclass
class TaskEnvironment:
    """Describes the environment a task needs."""
    docker_image: Optional[str] = None          # For B, C, E
    docker_compose: Optional[str] = None        # For E (multi-container)
    vm_image: Optional[str] = None              # For F
    vm_platform: Optional[str] = None           # "docker" | "aws" | "vmware"
    tool_servers: list[str] = field(default_factory=list)  # For E
    setup_script: Optional[str] = None
    teardown_script: Optional[str] = None
    eval_script: Optional[str] = None           # For F (per-task eval)


@dataclass
class ExecutionContext:
    """Runtime state during task execution."""
    task_id: str
    container_id: Optional[str] = None          # For B, C, E
    vm_id: Optional[str] = None                 # For F
    tool_server_ids: list[str] = field(default_factory=list)  # For E
    env_state: dict[str, Any] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    action_count: int = 0                       # For F efficiency scoring
    cost_usd: float = 0.0                       # Total cost across all models
    model_costs: dict[str, float] = field(default_factory=dict)  # Per-model cost breakdown
    model_tokens: dict[str, dict] = field(default_factory=dict)  # Per-model token usage
    # e.g. {"claude": {"input": 12000, "output": 3400}, "gpt-4o": {"input": 800, "output": 200}}


@dataclass
class GradeResult:
    """Result of grading a single submission."""
    task_id: str
    passed: bool
    score: float  # 0.0 - 1.0
    expected: str
    actual: str
    method: str  # Which grader matched (exact, normalized, numeric, fuzzy, etc.)
    details: dict[str, Any] | None = None


class Benchmark(ABC):
    """Base class for all benchmarks.

    Every benchmark declares its ``execution_mode`` and implements:
    - ``get_tasks()`` → list of Task objects
    - ``grade(task, result, context)`` → GradeResult

    Optionally override ``setup_environment`` / ``teardown_environment``
    for benchmarks that need Docker containers, VMs, or tool servers.
    """

    name: str = "unknown"
    description: str = ""
    execution_mode: ExecutionMode = ExecutionMode.DIRECT

    @abstractmethod
    def get_tasks(self) -> list[Task]:
        """Return all tasks for this benchmark."""
        ...

    @abstractmethod
    def grade(self, task: Task, result: Any, context: ExecutionContext) -> GradeResult:
        """Grade a single task result.

        Args:
            task: The Task object being graded.
            result: The agent's output (string for DIRECT, code for CODE_SUBMIT, etc.)
            context: Runtime state (container IDs, history, cost, etc.)

        Returns:
            GradeResult with pass/fail, score, and method info.
        """
        ...

    def setup_environment(self, task: Task) -> ExecutionContext:
        """Set up the execution environment for a task. Override for custom setup."""
        return ExecutionContext(task_id=task.id)

    def teardown_environment(self, context: ExecutionContext) -> None:
        """Clean up after task execution. Override for custom teardown."""
        pass

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def grade_legacy(self, task_id: str, submission: str) -> GradeResult:
        """Legacy signature for backward compatibility.

        Finds the task by ID, creates a minimal ExecutionContext, and
        delegates to the new ``grade(task, result, context)`` method.
        """
        tasks = self.get_tasks()
        task = next((t for t in tasks if t.id == task_id), None)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        context = ExecutionContext(task_id=task_id)
        return self.grade(task, submission, context)

    def grade_all(self, results: list[dict]) -> list[GradeResult]:
        """Grade all results from a run.

        Args:
            results: List of dicts with ``task_id`` and ``submission`` keys.
        """
        grades = []
        for r in results:
            if r.get("submission") is not None:
                grade = self.grade_legacy(r["task_id"], r["submission"])
                grades.append(grade)
        return grades

    def get_num_tasks(self) -> int:
        """Return number of tasks (may be expensive, override if needed)."""
        return len(self.get_tasks())

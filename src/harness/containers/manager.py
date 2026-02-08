"""Docker container lifecycle management for Terminal-Bench tasks.

This module bridges agent-harness with Terminal-Bench's Docker + tmux
infrastructure.  Rather than re-implementing container management, we wrap
Terminal-Bench's ``Terminal``, ``TmuxSession``, ``TaskPaths``, and
``TrialHandler`` classes directly.

Typical lifecycle::

    mgr = ContainerManager(output_dir=Path("./results/terminal-bench/run1"))
    session = mgr.start(task_id, task_dir)   # docker-compose up + tmux
    # ... agent sends commands via session ...
    passed, details = mgr.run_tests(task_id)  # copy tests, run, parse
    mgr.stop(task_id)                          # docker-compose down
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ContainerSession — agent-facing handle
# ---------------------------------------------------------------------------

@dataclass
class ContainerSession:
    """Lightweight handle for running commands inside a Docker container.

    Agents interact with this instead of the Docker API directly.
    Under the hood it delegates to Terminal-Bench's ``TmuxSession``.
    """

    task_id: str
    instruction: str = ""
    _tmux_session: Any = field(repr=False, default=None)
    _terminal: Any = field(repr=False, default=None)

    # -- public API for agents -----------------------------------------------

    def run(
        self,
        command: str,
        *,
        timeout: float = 180.0,
        block: bool = True,
    ) -> str:
        """Send a shell command and return the captured output.

        Args:
            command: The bash command to run.
            timeout: Maximum seconds to wait (when *block* is ``True``).
            block: If ``True``, wait for the command to finish.

        Returns:
            Captured terminal output (full scrollback).
        """
        if self._tmux_session is None:
            raise RuntimeError(
                "ContainerSession has no live tmux session. "
                "Was the container started?"
            )
        self._tmux_session.send_keys(
            [command, "Enter"],
            block=block,
            max_timeout_sec=timeout,
        )
        return self._tmux_session.capture_pane(capture_entire=True)

    def read_screen(self) -> str:
        """Capture the current visible terminal screen."""
        if self._tmux_session is None:
            return ""
        return self._tmux_session.capture_pane(capture_entire=False)

    def read_full(self) -> str:
        """Capture full scrollback buffer."""
        if self._tmux_session is None:
            return ""
        return self._tmux_session.capture_pane(capture_entire=True)

    def get_incremental(self) -> str:
        """Return output since the last incremental read."""
        if self._tmux_session is None:
            return ""
        return self._tmux_session.get_incremental_output()

    def copy_file_in(
        self,
        local_path: Path,
        container_dir: str | None = None,
    ) -> None:
        """Copy a local file into the running container."""
        if self._terminal is None:
            raise RuntimeError("No terminal available.")
        self._terminal.copy_to_container(
            paths=[local_path],
            container_dir=container_dir,
        )


# ---------------------------------------------------------------------------
# ContainerManager — owns the full lifecycle
# ---------------------------------------------------------------------------

class ContainerManager:
    """Manages Docker containers for Terminal-Bench tasks.

    Uses Terminal-Bench's ``Terminal`` / ``spin_up_terminal`` under the hood.

    Usage::

        mgr = ContainerManager(output_dir=run_output_dir)
        session = mgr.start(task_id, task_dir)
        # ... agent interacts via session.run() ...
        passed, details = mgr.run_tests(task_id)
        mgr.stop(task_id)
    """

    def __init__(self, output_dir: Path | None = None):
        self._output_dir = Path(output_dir) if output_dir else Path("./tb-runs")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # task_id → (Terminal, context-manager)
        self._terminals: dict[str, tuple[Any, Any]] = {}
        # task_id → ContainerSession
        self._sessions: dict[str, ContainerSession] = {}
        # task_id → task_dir (Path)
        self._task_dirs: dict[str, Path] = {}

    # -- lifecycle -----------------------------------------------------------

    def start(self, task_id: str, task_dir: Path) -> ContainerSession:
        """Build and start a container, return a ``ContainerSession``."""
        self._ensure_deps()

        from terminal_bench.terminal.terminal import spin_up_terminal
        from terminal_bench.handlers.trial_handler import TaskPaths

        task_paths = TaskPaths(task_dir)

        # Directories for terminal-bench logs
        trial_dir = self._output_dir / task_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        logs_path = trial_dir / "session-logs"
        logs_path.mkdir(exist_ok=True)
        agent_logs_path = trial_dir / "agent-logs"
        agent_logs_path.mkdir(exist_ok=True)

        # Unique names derived from task_id
        image_prefix = f"tb__{task_id}".replace(".", "-")
        container_name = f"harness-{task_id}".replace(".", "-")
        image_name = f"{image_prefix}__client"

        ctx = spin_up_terminal(
            client_container_name=container_name,
            client_image_name=image_name,
            docker_compose_path=task_paths.docker_compose_path,
            docker_image_name_prefix=image_prefix,
            sessions_logs_path=logs_path,
            agent_logs_path=agent_logs_path,
            disable_recording=True,
        )
        terminal = ctx.__enter__()
        self._terminals[task_id] = (terminal, ctx)
        self._task_dirs[task_id] = task_dir

        # Create the agent tmux session in the container
        tmux_session = terminal.create_session("agent")

        # Load instruction from task.yaml
        instruction = ""
        try:
            from terminal_bench.handlers.trial_handler import Task as TBTask
            tb_task = TBTask.from_yaml(task_paths.task_config_path)
            instruction = tb_task.instruction
        except Exception:
            pass

        session = ContainerSession(
            task_id=task_id,
            instruction=instruction,
            _tmux_session=tmux_session,
            _terminal=terminal,
        )
        self._sessions[task_id] = session
        return session

    def run_tests(
        self,
        task_id: str,
        *,
        timeout: float = 120.0,
    ) -> tuple[bool, dict[str, Any]]:
        """Run the task's test suite inside the container.

        Follows the same flow as Terminal-Bench's ``Harness._run_tests``:
        1. Copy ``run-tests.sh`` and ``tests/`` into ``/tests`` inside the
           container.
        2. Execute ``bash /tests/run-tests.sh``.
        3. Parse the output with the task's configured parser.

        Returns:
            ``(passed, details)`` — *details* includes test output, per-test
            results, and the parser name.
        """
        pair = self._terminals.get(task_id)
        task_dir = self._task_dirs.get(task_id)
        if pair is None or task_dir is None:
            return False, {"error": "container not running"}

        terminal, _ = pair

        try:
            from terminal_bench.handlers.trial_handler import (
                TaskPaths,
                Task as TBTask,
            )
            from terminal_bench.terminal.docker_compose_manager import (
                DockerComposeManager,
            )
            from terminal_bench.parsers.parser_factory import ParserFactory
            from terminal_bench.parsers.base_parser import UnitTestStatus

            task_paths = TaskPaths(task_dir)

            # Load task config for parser selection and test timeout
            tb_task = TBTask.from_yaml(task_paths.task_config_path)
            parser = ParserFactory.get_parser(tb_task.parser_name)
            test_timeout = min(timeout, tb_task.max_test_timeout_sec)

            # 1. Copy tests into container at /tests
            paths_to_copy = [task_paths.run_tests_path]
            if task_paths.test_dir.exists():
                paths_to_copy.append(task_paths.test_dir)
            terminal.copy_to_container(
                paths=paths_to_copy,
                container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
            )

            # 2. Run tests in a dedicated session
            test_session = terminal.create_session("tests")
            test_cmd = (
                f"bash {DockerComposeManager.CONTAINER_TEST_DIR}"
                f"/{task_paths.run_tests_path.name}"
            )
            test_session.send_keys(
                [test_cmd, "Enter"],
                block=True,
                max_timeout_sec=test_timeout,
            )

            # 3. Capture output and parse
            test_output = test_session.capture_pane(capture_entire=True)
            results = parser.parse(test_output)

            passed = all(
                v == UnitTestStatus.PASSED for v in results.values()
            )

            return passed, {
                "test_output": test_output[-4000:],  # truncate for storage
                "test_results": {k: v.value for k, v in results.items()},
                "parser": tb_task.parser_name.value,
                "num_passed": sum(
                    1 for v in results.values()
                    if v == UnitTestStatus.PASSED
                ),
                "num_total": len(results),
            }

        except TimeoutError:
            return False, {"error": "test execution timed out"}
        except Exception as e:
            return False, {"error": str(e)}

    def stop(self, task_id: str) -> None:
        """Stop and remove the container for the given task."""
        pair = self._terminals.pop(task_id, None)
        if pair is not None:
            _, ctx = pair
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        self._sessions.pop(task_id, None)
        self._task_dirs.pop(task_id, None)

    def stop_all(self) -> None:
        """Stop all running containers."""
        for task_id in list(self._terminals):
            self.stop(task_id)

    def get_session(self, task_id: str) -> ContainerSession | None:
        """Retrieve an existing session for a running container."""
        return self._sessions.get(task_id)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _ensure_deps() -> None:
        """Check that terminal-bench and Docker are available."""
        try:
            import terminal_bench  # noqa: F401
        except ImportError:
            raise ImportError(
                "terminal-bench is required for Terminal-Bench tasks. "
                "Install it with:  pip install terminal-bench"
            )

        if shutil.which("docker") is None:
            raise RuntimeError(
                "Docker is required for Terminal-Bench tasks but was not "
                "found on PATH.  Install Docker: "
                "https://docs.docker.com/get-docker/"
            )

    @staticmethod
    def docker_available() -> bool:
        """Quick check: is Docker installed and responsive?"""
        if shutil.which("docker") is None:
            return False
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

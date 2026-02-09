"""Agent base class with mode-specific methods.

Agents are *mode-aware, not benchmark-aware*.  A single
``run_interactive_task()`` implementation works for Terminal-Bench and
CORE-Bench; a single ``run_task()`` works for GAIA, HLE, BrowseComp, etc.

Subclass ``AgentBase`` and implement the methods corresponding to the
execution modes your agent supports.  All methods raise
``NotImplementedError`` by default so that agents can be incrementally
extended without implementing every mode upfront.
"""

from abc import ABC
from typing import Any, Callable


class AgentBase(ABC):
    """Unified agent protocol — one method per execution archetype.

    All methods raise ``NotImplementedError`` rather than being
    ``@abstractmethod`` so that an agent can implement only the modes
    it cares about.  The ``TaskOrchestrator`` checks capability at
    dispatch time and raises a clear error if the agent doesn't
    support the required mode.
    """

    # ── Archetype A: DIRECT ──────────────────────────────────────
    def run_task(self, task_id: str, task_data: dict) -> str:
        """Receive a question, return an answer string.

        Used by: GAIA, AssistantBench, HLE, ARC-AGI 1 & 2, BrowseComp
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_task() (DIRECT mode)"
        )

    # ── Archetype B: CODE_SUBMIT ─────────────────────────────────
    def run_code_task(self, task_id: str, task_data: dict) -> str:
        """Receive a problem, return code (patch, script, or solution).

        Used by: SWE-bench, USACO, SciCode, ScienceAgentBench
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_code_task() (CODE_SUBMIT mode)"
        )

    # ── Archetype C: INTERACTIVE ─────────────────────────────────
    def run_interactive_task(
        self,
        task_id: str,
        task_data: dict,
        exec_fn: Callable[[str], str],      # exec_fn("ls -la") → stdout
    ) -> None:
        """Run commands iteratively in a live container.

        Args:
            exec_fn: Callable that sends a shell command and returns stdout.

        Used by: Terminal-Bench, CORE-Bench
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_interactive_task() (INTERACTIVE mode)"
        )

    # ── Archetype D: CONVERSATIONAL ──────────────────────────────
    def run_conversational_task(
        self,
        task_id: str,
        task_data: dict,
        send_fn: Callable[[str], str],       # send message → user response
        tool_fn: Callable[[str, dict], Any],  # call domain tool → result
    ) -> None:
        """Converse with a simulated user, optionally calling tools.

        Args:
            send_fn: Send a message to the simulated user, get their response.
            tool_fn: Call a domain-specific tool (e.g. booking API).

        Used by: TAU-bench
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_conversational_task() (CONVERSATIONAL mode)"
        )

    # ── Archetype E: TOOL_USE ────────────────────────────────────
    def run_tool_task(
        self,
        task_id: str,
        task_data: dict,
        tool_fn: Callable[[str, str, dict], Any],
        # tool_fn(server_name, tool_name, params) → result
    ) -> str:
        """Discover and orchestrate calls to live tool servers.

        Returns the final answer string for claims-based grading.

        Args:
            tool_fn: Call a tool on a named server.
                     ``tool_fn("exa", "search", {"query": "..."})``

        Used by: MCP-Atlas, Finance Agent
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_tool_task() (TOOL_USE mode)"
        )

    # ── Archetype F: GUI_AGENT ───────────────────────────────────
    def run_gui_task(
        self,
        task_id: str,
        task_data: dict,
        observe_fn: Callable[[], dict],
        # observe_fn() → {"screenshot": bytes, "a11y_tree": str, ...}
        act_fn: Callable[[dict], dict],
        # act_fn({"type": "click", "x": 100, "y": 200}) → result
    ) -> None:
        """Observe a VM/game environment and take actions.

        Loop: observe → decide → act → repeat until done.

        Args:
            observe_fn: Capture the current state (screenshot, a11y tree, etc.)
            act_fn: Execute an action (click, type, key press, drag, etc.)

        Used by: OSWorld, GDPval-AA, ARC-AGI-3
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_gui_task() (GUI_AGENT mode)"
        )

    # ── Capability check ─────────────────────────────────────────

    _MODE_METHODS = {
        "direct": "run_task",
        "code_submit": "run_code_task",
        "interactive": "run_interactive_task",
        "conversational": "run_conversational_task",
        "tool_use": "run_tool_task",
        "gui_agent": "run_gui_task",
    }

    def supports_mode(self, mode: str) -> bool:
        """Check whether the agent has overridden the method for *mode*.

        ``mode`` should be the ``ExecutionMode.value`` string
        (e.g. ``"direct"``, ``"interactive"``).
        """
        method_name = self._MODE_METHODS.get(mode)
        if method_name is None:
            return False
        method = getattr(type(self), method_name, None)
        base_method = getattr(AgentBase, method_name, None)
        return method is not base_method

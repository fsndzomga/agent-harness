#!/usr/bin/env python3
"""
Terminal-Bench agent — LLM-in-the-loop shell interaction.

This agent handles Terminal-Bench tasks by:
  1. Starting a Docker container for the task
  2. Reading the task instruction
  3. Running an LLM-in-the-loop where the model issues shell commands
  4. Running the task's test suite inside the container
  5. Returning PASS/FAIL as the submission

The agent manages the full container lifecycle.  After the test suite runs
(or the iteration limit is reached), the container is stopped.

Configuration (via environment variables):
  HARNESS_MODEL            — LLM model to use (required)
  TB_MAX_ITERATIONS        — Maximum command iterations (default: 30)
  TB_COMMAND_TIMEOUT       — Per-command timeout in seconds (default: 120)
  TB_TEST_TIMEOUT          — Test suite timeout in seconds (default: 120)

Usage:
  harness run --agent agents/terminal_agent.py \\
              --benchmark terminal-bench \\
              --model openrouter/deepseek/deepseek-chat-v3-0324 \\
              --task-timeout 600 \\
              --parallel 2
"""

import sys
import os
import json

# Add parent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pathlib import Path
from harness.agent import Agent
from harness.providers.base import Message


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a skilled DevOps / systems engineer working inside a Linux terminal.
You will be given a task to accomplish.  You interact with the terminal by
issuing ONE shell command at a time and observing the output.

IMPORTANT CONTEXT:
- You are ALREADY inside a pre-configured Docker container for this task.
- All necessary files, repos, and tools are ALREADY present in the container.
- Start by exploring the environment: run `pwd`, `ls`, `find / -maxdepth 3 -type f 2>/dev/null | head -40`, etc.
- Do NOT try to clone repos from external URLs — the files are local.
- Do NOT try to access the internet unless the task explicitly provides a working URL.
- When a task mentions a repository, it is already cloned/initialized locally.

Rules:
- Issue exactly ONE command per turn (no && chaining, no ; chaining).
- After each command, WAIT for the output before deciding the next step.
- Think step-by-step.  If a command fails, debug and retry.
- ALWAYS start by exploring: `pwd`, `ls -la`, look at existing files.
- When you believe the task is COMPLETE, respond with exactly: DONE
- Be efficient — avoid unnecessary commands.
- Do NOT use interactive editors (vim, nano).  Use sed, awk, tee, cat <<EOF, or echo.
- For file creation, prefer `cat > file <<'EOF'` patterns.
- When installing packages, use non-interactive flags (apt-get -y, pip -q, etc.).
- If a command hangs or times out, do NOT retry the same command — try a different approach.

Respond with ONLY the command to execute (or DONE).  No explanation, no markdown fences.\
"""


class TerminalAgent(Agent):
    """LLM-driven terminal agent for Terminal-Bench tasks."""

    def __init__(self):
        super().__init__()
        self.max_iterations = int(os.environ.get("TB_MAX_ITERATIONS", "30"))
        self.command_timeout = float(os.environ.get("TB_COMMAND_TIMEOUT", "1220"))
        self.test_timeout = float(os.environ.get("TB_TEST_TIMEOUT", "120"))

    def run_task(self, task_id: str, task_data: dict) -> str:
        """Execute a terminal-bench task.

        For terminal tasks (task_type == "terminal"), the agent starts a
        Docker container, runs an LLM loop, then grades via the test suite.

        For non-terminal tasks, falls back to single-shot QA.
        """
        task_type = task_data.get("task_type", "")
        if task_type != "terminal":
            return self._fallback_qa(task_data)

        instruction = task_data.get("instruction", "")
        task_dir = task_data.get("task_dir", "")

        if not task_dir:
            return "FAIL"

        self.log(type="info", content=f"Terminal task: {instruction[:120]}...")
        self.increment("terminal_tasks")

        # Import container manager (requires terminal-bench + Docker)
        from harness.containers import ContainerManager

        mgr = ContainerManager(
            output_dir=Path(f"/tmp/tb-agent-{task_id}"),
        )

        try:
            # 1. Start container
            self.log(type="info", content="Starting container...")
            session = mgr.start(task_id, Path(task_dir))
            self.increment("containers_started")

            # 2. LLM-in-the-loop terminal interaction
            passed_interaction = self._interact(
                session, instruction, task_id,
            )

            # 3. Run the test suite
            self.log(type="info", content="Running tests...")
            passed, details = mgr.run_tests(
                task_id, timeout=self.test_timeout,
            )
            self.metric("tests_passed", passed)

            if details.get("num_total"):
                self.metric("test_count", details["num_total"])
                self.metric(
                    "test_pass_count", details.get("num_passed", 0),
                )

            # Log test details
            self.log(
                type="test_results",
                passed=passed,
                details={
                    k: v
                    for k, v in details.items()
                    if k != "test_output"
                },
            )

            submission = "PASS" if passed else "FAIL"
            return submission

        except Exception as exc:
            self.log(type="error", content=str(exc))
            return "FAIL"

        finally:
            # Always clean up
            try:
                mgr.stop_all()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # LLM-in-the-loop shell interaction
    # ------------------------------------------------------------------

    def _interact(
        self,
        session,
        instruction: str,
        task_id: str,
    ) -> bool:
        """Drive the LLM interaction loop.

        Returns ``True`` if the agent signaled DONE, ``False`` if we hit
        the iteration limit.
        """
        # Capture initial environment for the LLM
        try:
            env_snapshot = session.run("pwd && ls -la", timeout=10)
        except Exception:
            env_snapshot = "(could not capture initial state)"

        messages: list[Message] = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    f"TASK:\n{instruction}\n\n"
                    f"CURRENT ENVIRONMENT:\n```\n{env_snapshot}\n```\n\n"
                    "The terminal is ready inside the Docker container. "
                    "All task files are already present locally. "
                    "Start by exploring what's here, then work on the task. "
                    "Issue your first command."
                ),
            ),
        ]

        for i in range(self.max_iterations):
            self.increment("iterations")

            # Get next command from LLM
            response = self.complete(messages, temperature=0.2)
            raw_reply = response.message.content.strip()

            # Strip markdown fences if the model wraps the command
            command = self._extract_command(raw_reply)

            # Check for completion signal
            if command.upper() == "DONE":
                self.log(
                    type="info",
                    content=f"Agent signaled DONE after {i + 1} iterations",
                )
                self.metric("iterations_to_done", i + 1)
                return True

            # Execute command in container
            self.log(type="command", iteration=i + 1, command=command)
            self.record_tool_use("shell_exec")

            try:
                output = session.run(
                    command, timeout=self.command_timeout,
                )
            except Exception as exc:
                output = f"[ERROR executing command: {exc}]"

            # Truncate long output for context window management
            if len(output) > 8000:
                output = (
                    output[:3000]
                    + "\n\n... [output truncated] ...\n\n"
                    + output[-3000:]
                )

            self.log(
                type="output",
                iteration=i + 1,
                output=output[-2000:],
            )

            # Feed output back to LLM
            messages.append(Message(role="assistant", content=raw_reply))
            messages.append(
                Message(
                    role="user",
                    content=(
                        f"OUTPUT:\n```\n{output}\n```\n\n"
                        "Next command (or DONE if finished):"
                    ),
                )
            )

            # Compact history if it's getting long (keep system + last N)
            if len(messages) > 40:
                messages = self._compact_history(messages)

        self.log(
            type="warning",
            content=f"Hit iteration limit ({self.max_iterations})",
        )
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_command(text: str) -> str:
        """Extract a shell command from LLM output.

        Handles common LLM quirks like wrapping in code fences.
        """
        text = text.strip()

        # Remove markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```bash or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # If multiple lines, take only the first non-empty one
        # (agent should issue one command at a time)
        for line in text.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                return line

        return text

    @staticmethod
    def _compact_history(messages: list[Message]) -> list[Message]:
        """Keep the system prompt, first user message, and recent turns."""
        system = messages[0]
        first_user = messages[1]
        # Keep the last 20 messages (10 turns)
        recent = messages[-20:]
        return [system, first_user] + recent

    def _fallback_qa(self, task_data: dict) -> str:
        """Simple single-shot QA for non-terminal tasks."""
        question = task_data.get("question", task_data.get("instruction", str(task_data)))
        response = self.complete([
            Message(
                role="system",
                content="Answer the question concisely and directly.",
            ),
            Message(role="user", content=question),
        ])
        return response.message.content


if __name__ == "__main__":
    TerminalAgent().run()

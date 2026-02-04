"""Agent runner - executes agents as subprocesses."""

import subprocess
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .protocol import Task, Submission

if TYPE_CHECKING:
    from .logging import TraceLogger


class AgentRunner:
    """Runs an agent on a single task via subprocess."""
    
    def __init__(
        self,
        agent_path: Path,
        logger: "TraceLogger | None" = None,
        env: dict[str, str] | None = None,
    ):
        self.agent_path = Path(agent_path).resolve()  # Convert to absolute path
        self.logger = logger
        self.env = env or {}
        self.command = self._detect_command()
    
    def _detect_command(self) -> list[str]:
        """Figure out how to run the agent."""
        p = self.agent_path
        
        # Check for manifest.yaml first
        if p.is_dir():
            manifest_path = p / "manifest.yaml"
            if manifest_path.exists():
                try:
                    import yaml
                    manifest = yaml.safe_load(manifest_path.read_text())
                    if "run" in manifest:
                        return manifest["run"].split()
                except ImportError:
                    pass  # yaml not installed, skip manifest
        
        if p.is_file():
            if p.suffix == ".py":
                return ["python", str(p)]
            elif p.suffix == ".sh":
                return ["bash", str(p)]
            else:
                return [str(p)]  # Assume executable
        
        # Directory: look for entry points
        if (p / "agent.py").exists():
            return ["python", str(p / "agent.py")]
        if (p / "agent.sh").exists():
            return ["bash", str(p / "agent.sh")]
        if (p / "agent").exists():
            return [str(p / "agent")]
        if (p / "__main__.py").exists():
            return ["python", str(p)]
        
        raise ValueError(f"Cannot determine how to run agent at {p}")
    
    def run(self, task: Task, timeout: float = 300) -> Submission:
        """Run agent on a single task, return submission."""
        
        start_time = time.time()
        
        if self.logger:
            self.logger.log("task_start", task_id=task.id, task_data=task.data)
        
        # Prepare environment
        run_env = {**os.environ, **self.env}
        run_env["HARNESS_TASK_ID"] = task.id
        
        proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.agent_path if self.agent_path.is_dir() else self.agent_path.parent,
            env=run_env,
        )
        
        # Send task
        request_line = task.to_jsonrpc()
        
        if self.logger:
            self.logger.log("task_sent", task_id=task.id)
        
        # Send task and read output (with timeout)
        try:
            stdout, stderr = proc.communicate(input=request_line + "\n", timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()  # Clean up
            if self.logger:
                self.logger.log("task_timeout", task_id=task.id, timeout=timeout)
            raise TimeoutError(f"Agent timed out after {timeout}s")
        
        # Process output lines
        submission = None
        
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # Non-JSON output, log as debug
                if self.logger:
                    self.logger.log("agent_stdout", content=line)
                continue
            
            # Agent logging (completions, tool calls, etc.)
            if msg.get("method") == "log":
                if self.logger:
                    self.logger.log_raw(msg.get("params", {}))
            
            # Final result
            elif "result" in msg:
                submission = Submission.from_jsonrpc(line)
                duration_ms = int((time.time() - start_time) * 1000)
                if self.logger:
                    self.logger.log(
                        "task_complete",
                        task_id=task.id,
                        submission=submission.answer,
                        duration_ms=duration_ms,
                    )
        
        if proc.returncode != 0:
            error_msg = f"Agent failed (exit {proc.returncode}): {stderr}"
            if self.logger:
                self.logger.log("task_error", task_id=task.id, error=error_msg, stderr=stderr)
            raise RuntimeError(error_msg)
        
        if submission is None:
            error_msg = f"No valid submission in output:\n{stdout}"
            if self.logger:
                self.logger.log("task_error", task_id=task.id, error=error_msg)
            raise RuntimeError(error_msg)
        
        return submission

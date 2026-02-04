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
        
        # File extension mapping
        ext_commands = {
            ".py": ["python"],
            ".sh": ["bash"],
            ".rb": ["ruby"],
            ".js": ["node"],
            ".ts": ["npx", "ts-node"],
            ".mjs": ["node"],
            ".pl": ["perl"],
            ".php": ["php"],
            ".lua": ["lua"],
            ".r": ["Rscript"],
            ".R": ["Rscript"],
            ".jl": ["julia"],
        }
        
        if p.is_file():
            if p.suffix in ext_commands:
                return ext_commands[p.suffix] + [str(p)]
            else:
                return [str(p)]  # Assume executable (Go binary, Rust binary, etc.)
        
        # Directory: look for entry points (in priority order)
        entry_points = [
            ("agent.py", ["python"]),
            ("agent.rb", ["ruby"]),
            ("agent.js", ["node"]),
            ("agent.ts", ["npx", "ts-node"]),
            ("agent.sh", ["bash"]),
            ("agent.pl", ["perl"]),
            ("agent.php", ["php"]),
            ("agent.lua", ["lua"]),
            ("agent.jl", ["julia"]),
            ("main.go", ["go", "run"]),
            ("agent", []),  # Compiled binary
            ("__main__.py", ["python"]),
        ]
        
        for filename, cmd in entry_points:
            entry = p / filename
            if entry.exists():
                if cmd:
                    return cmd + [str(entry)]
                else:
                    return [str(entry)]
        
        # Check for project files that indicate how to run
        if (p / "Cargo.toml").exists():
            return ["cargo", "run", "--manifest-path", str(p / "Cargo.toml"), "--"]
        if (p / "go.mod").exists() and (p / "main.go").exists():
            return ["go", "run", str(p)]
        if (p / "package.json").exists():
            # Check for start script or main entry
            return ["npm", "start", "--prefix", str(p)]
        if (p / "Gemfile").exists() and (p / "agent.rb").exists():
            return ["bundle", "exec", "ruby", str(p / "agent.rb")]
        
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

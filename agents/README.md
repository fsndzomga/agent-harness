# Agents

This directory contains agents that can be evaluated using the harness.

## Convention

- **Single-file agent**: `agents/my_agent.py` — no extra deps beyond the harness
- **Module agent**: `agents/my_agent/` directory with:
  - `agent.py` — entry point (runs on `python agent.py`)
  - `requirements.txt` — agent-specific dependencies
  - Any other supporting files (prompts, tools, configs)

## Available Agents

| Agent | Type | Description |
|-------|------|-------------|
| `echo_agent.py` | Single-file | Echoes task data back (for testing) |
| `simple_qa_agent.py` | Single-file | Single LLM call, no tools |
| `metrics_agent.py` | Single-file | Demonstrates metrics tracking |
| `hal_generalist/` | Module | smolagents CodeAgent with web search + file tools (GAIA) |

## Usage

```bash
# Single-file agent
harness run --agent agents/echo_agent.py --benchmark arithmetic

# Module agent
harness run --agent agents/hal_generalist --benchmark gaia-level1 --model openrouter/deepseek/deepseek-chat-v3-0324

# Install module agent deps first
pip install -r agents/hal_generalist/requirements.txt
```

## Writing Your Own Agent

### Single-File (Python)

```python
from harness.agent import Agent
from harness.providers.base import Message

class MyAgent(Agent):
    def run_task(self, task_id: str, task_data: dict) -> str:
        response = self.complete([
            Message(role="user", content=task_data["question"])
        ])
        return response.message.content

if __name__ == "__main__":
    MyAgent().run()
```

### Any Language

Just read JSON-RPC from stdin and write responses to stdout:

```bash
harness run --agent ./my_agent.sh --benchmark arithmetic
```

See the main [README](../README.md) for the full protocol spec.

# Agent Harness

A local-first, lightweight harness for AI agent evaluations.

## Features

- **Local-first**: No Weave API key required. All logs are JSONL files you can inspect directly.
- **Language-agnostic agents**: Write agents in Python, Rust, Go, Node, bash - anything that can do stdin/stdout JSON.
- **Automatic logging**: Raw LLM API requests/responses are captured automatically - no manual logging code needed.
- **Parallel execution**: Run tasks in parallel with automatic retry for API errors (429, 5xx, timeouts).
- **Layered grading**: Exact → normalized → numeric → fuzzy matching, with LLM-as-judge support (coming soon).
- **Benchmark plugins**: Built-in support for GAIA, with more coming.

## Quick Start

```bash
# Install with Poetry
cd agent-harness
poetry install

# Or install with pip
pip install -e .

# Set your API key
export ANTHROPIC_API_KEY="sk-..."
# or
export OPENAI_API_KEY="sk-..."

# Run the simple QA agent on the arithmetic benchmark
harness run \
    --agent src/harness/examples/simple_qa_agent.py \
    --benchmark arithmetic \
    --output ./results \
    --num-tasks 10

# View results
harness view ./results
```

## Agent Protocol

Agents communicate via stdin/stdout using JSON-RPC:

```
Harness → Agent (stdin):
{"jsonrpc": "2.0", "method": "run_task", "params": {"task_id": "abc", "task_data": {...}}, "id": 1}

Agent → Harness (stdout):
{"jsonrpc": "2.0", "result": {"task_id": "abc", "submission": "42"}, "id": 1}
```

Agents can emit logs that get captured automatically:
```
{"jsonrpc": "2.0", "method": "log", "params": {"type": "thinking", "content": "..."}}
```

### Python Agent Helper

For Python agents, use the `Agent` base class:

```python
from harness.agent import Agent
from harness.providers.base import Message

class MyAgent(Agent):
    def run_task(self, task_id: str, task_data: dict) -> str:
        # LLM calls are automatically logged
        response = self.complete([
            Message(role="user", content=task_data["question"])
        ])
        return response.message.content

if __name__ == "__main__":
    MyAgent().run()
```

### Any Language

Write agents in any language - just implement the JSON-RPC protocol:

```bash
#!/bin/bash
read line
task_id=$(echo "$line" | jq -r '.params.task_id')
echo "{\"jsonrpc\": \"2.0\", \"result\": {\"task_id\": \"$task_id\", \"submission\": \"hello\"}, \"id\": 1}"
```

## Benchmarks

### Available Benchmarks

- `arithmetic` - Simple arithmetic problems (built-in, no dependencies)
- `gaia` - GAIA benchmark (requires `datasets` package)
- `gaia-level1`, `gaia-level2`, `gaia-level3` - GAIA by difficulty level

```bash
# List available benchmarks
harness benchmarks

# Run GAIA Level 1
pip install datasets  # If not installed
harness run --agent ./my_agent --benchmark gaia-level1 --output ./results
```

### Adding Benchmarks

Create a class that inherits from `Benchmark`:

```python
from harness.benchmarks.base import Benchmark, GradeResult
from harness.protocol import Task

class MyBenchmark(Benchmark):
    name = "my-benchmark"
    description = "My custom benchmark"
    
    def get_tasks(self) -> list[Task]:
        return [Task(id="t1", data={"question": "What is 2+2?"})]
    
    def grade(self, task_id: str, submission: str) -> GradeResult:
        expected = "4"
        passed = submission.strip() == expected
        return GradeResult(
            task_id=task_id,
            passed=passed,
            score=1.0 if passed else 0.0,
            expected=expected,
            actual=submission,
            method="exact" if passed else "none",
        )
```

## Grading

The harness supports multiple grading modes controlled via the `--grader` option.

### Available Graders

| Grader | Description |
|--------|-------------|
| `exact` | Exact string match after trimming whitespace |
| `normalized` | Match after lowercase + whitespace normalization |
| `numeric` | Match numeric values with tolerance (±0.1%) |
| `contains` | Check if expected answer is contained in submission |
| `fuzzy` | Fuzzy string match (90% similarity threshold) |
| `strict` | Only exact or normalized match |
| `default` | Try all graders: exact → normalized → numeric → contains → fuzzy |
| `llm` | Use LLM-as-judge for semantic evaluation |
| `llm-fallback` | Try deterministic graders first, fall back to LLM |

### Examples

```bash
# Strict grading - exact or normalized match only
harness run --agent ./agent.py -b gaia-level1 -o results -g strict

# LLM-as-judge for all grading
harness run --agent ./agent.py -b gaia-level1 -o results -g llm --model openrouter/anthropic/claude-sonnet-4-5-20250514

# Deterministic first, then LLM fallback (recommended)
harness run --agent ./agent.py -b gaia-level1 -o results -g llm-fallback --grader-model openrouter/anthropic/claude-sonnet-4-5-20250514
```

### Grading Behavior

The `default` grading pipeline tries matchers from strictest to most lenient:

1. **Exact match** - "42" == "42"
2. **Normalized match** - "The Answer" == "the answer"
3. **Numeric match** - "2.500" ≈ "2.5" (within 0.1%)
4. **Contains match** - "The answer is 42" contains "42"
5. **Fuzzy match** - "colour" ≈ "color" (90% similar)

For LLM-as-judge (`llm` or `llm-fallback`), the harness asks the LLM to evaluate semantic equivalence.

## Logging

All logs are JSONL files with raw API request/response data:

```json
{"timestamp": 1234567890.123, "type": "completion", "provider": "litellm/anthropic", "request": {"model": "claude-sonnet-4-5-20250514", "messages": [...]}, "response": {"id": "msg_...", "content": [...]}, "latency_ms": 1523}
```

View logs with any JSON tool:
```bash
cat ./results/trace_task1.jsonl | jq .
```

## CLI Reference

```bash
# Run single task
harness run-one --agent ./agent.py --task '{"id": "t1", "data": {"question": "2+2?"}}'

# Run benchmark
harness run --agent ./agent.py --benchmark gaia --output ./results

# Run with options
harness run \
    --agent ./agent.py \
    --benchmark gaia-level1 \
    --output ./results \
    --parallel 10 \
    --max-retries 3 \
    --task-timeout 300 \
    --num-tasks 50 \
    --model gpt-4o \
    --grader llm-fallback

# List benchmarks
harness benchmarks

# View results
harness view ./results
```

## Configuration

Set model via environment variable or CLI:

```bash
export HARNESS_MODEL="claude-sonnet-4-5-20250514"
# or
harness run --model gpt-4o ...
```

API keys are read from standard environment variables:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- etc. (via LiteLLM)

## Development

```bash
# Install with dev dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Run a quick test
poetry run harness run-one \
    --agent src/harness/examples/echo_agent.py \
    --task '{"id": "test", "data": {"x": 1}}'
```

## Roadmap

- [x] M1: Agent protocol + single task runner
- [x] M2: JSONL logging with raw API capture
- [x] M3: LiteLLM provider with auto-logging
- [x] M4: Parallel runner with retry logic
- [x] M5: Benchmark system with GAIA
- [ ] M6: Sandbox tiers (venv, firejail, docker)
- [ ] M7: HuggingFace upload + better viewer
- [ ] LLM-as-judge grading layer
- [ ] More benchmarks (SWE-bench, GPQA, etc.)

## License

MIT

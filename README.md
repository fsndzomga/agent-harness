# Agent Harness

A local-first, lightweight harness for AI agent evaluations.

## Features

- **Local-first**: No Weave API key required. All logs are JSONL files you can inspect directly.
- **Language-agnostic agents**: Write agents in Python, Rust, Go, Node, bash - anything that can do stdin/stdout JSON.
- **Automatic logging**: Raw LLM API requests/responses are captured automatically - no manual logging code needed.
- **Parallel execution**: Run tasks in parallel with automatic retry for API errors (429, 5xx, timeouts).
- **Layered grading**: Exact → normalized → numeric → fuzzy matching, with LLM-as-judge support.
- **Run metadata**: Comprehensive `run.json` with token usage, costs, latencies, and custom agent metrics.
- **Agent metrics**: Track custom KPIs (steps, tool usage, etc.) that get aggregated across runs.
- **Benchmark plugins**: Built-in support for GAIA and Terminal-Bench, with more coming.
- **Container-graded tasks**: Terminal-Bench integration with Docker-based task environments and automatic test-suite grading.

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
# Output goes to ./results/arithmetic/{run_id}/
harness run \
    --agent agents/simple_qa_agent.py \
    --benchmark arithmetic \
    --num-tasks 10

# View results
harness view ./results/arithmetic
```

## Agent Protocol

Agents communicate via stdin/stdout using JSON-RPC:

```
Harness → Agent (stdin):
{"jsonrpc": "2.0", "method": "run_task", "params": {"task_id": "abc", "task_data": {...}}, "id": 1}

Agent → Harness (stdout):
{"jsonrpc": "2.0", "result": {"task_id": "abc", "submission": "42", "metrics": {"steps": 3}}, "id": 1}
```

Agents can emit logs that get captured automatically:
```
{"jsonrpc": "2.0", "method": "log", "params": {"type": "thinking", "content": "..."}}
```

The optional `metrics` field in the response is used to report agent-specific KPIs.
```

### Python Agent Helper

For Python agents, use the `Agent` base class:

```python
from harness.agent import Agent
from harness.providers.base import Message

class MyAgent(Agent):
    def run_task(self, task_id: str, task_data: dict) -> str:
        self.increment("steps")  # Track metrics
        
        # LLM calls are automatically logged
        response = self.complete([
            Message(role="user", content=task_data["question"])
        ])
        
        self.record_tool_use("llm_call")  # Track tool usage
        return response.message.content

if __name__ == "__main__":
    MyAgent().run()
```

### Any Language

Write agents in any language - the harness auto-detects how to run them based on file extension or project structure.

#### Supported Languages

| Language | File Extension | Directory Entry | Project Detection |
|----------|---------------|-----------------|-------------------|
| Python | `.py` | `agent.py`, `__main__.py` | `.venv/` or `venv/` → uses venv python |
| Ruby | `.rb` | `agent.rb` | `Gemfile` → `bundle exec` |
| JavaScript | `.js`, `.mjs` | `agent.js` | `package.json` → `npm start` |
| TypeScript | `.ts` | `agent.ts` | - |
| Bash | `.sh` | `agent.sh` | - |
| Perl | `.pl` | `agent.pl` | - |
| PHP | `.php` | `agent.php` | - |
| Lua | `.lua` | `agent.lua` | - |
| Julia | `.jl` | `agent.jl` | - |
| R | `.r`, `.R` | - | - |
| Go | - | `main.go` | `go.mod` → `go run` |
| Rust | - | - | `Cargo.toml` → `cargo run` |
| Any compiled | - | `agent` (binary) | - |

#### Examples

**Ruby agent** (`agent.rb`):
```ruby
#!/usr/bin/env ruby
require 'json'

ARGF.each_line do |line|
  msg = JSON.parse(line)
  if msg["method"] == "run_task"
    task_id = msg["params"]["task_id"]
    question = msg["params"]["task_data"]["question"]
    
    # Your agent logic here
    answer = "42"
    
    result = { jsonrpc: "2.0", id: msg["id"], result: { task_id: task_id, submission: answer } }
    puts result.to_json
  end
end
```

**Node.js agent** (`agent.js`):
```javascript
const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin });

rl.on('line', (line) => {
  const msg = JSON.parse(line);
  if (msg.method === 'run_task') {
    const { task_id, task_data } = msg.params;
    
    // Your agent logic here
    const answer = "42";
    
    console.log(JSON.stringify({
      jsonrpc: "2.0",
      id: msg.id,
      result: { task_id, submission: answer }
    }));
  }
});
```

**Bash agent** (`agent.sh`):
```bash
#!/bin/bash
read line
task_id=$(echo "$line" | jq -r '.params.task_id')
msg_id=$(echo "$line" | jq -r '.id')
echo "{\"jsonrpc\": \"2.0\", \"result\": {\"task_id\": \"$task_id\", \"submission\": \"hello\"}, \"id\": $msg_id}"
```

**Rust agent** (with `Cargo.toml`):
```bash
# Directory structure:
# my-rust-agent/
#   Cargo.toml
#   src/main.rs

harness run --agent ./my-rust-agent --benchmark arithmetic
# Runs: cargo run --manifest-path ./my-rust-agent/Cargo.toml --
```

**Custom run command** (`manifest.yaml`):
```yaml
# my-agent/manifest.yaml
run: python -m my_custom_module
# or: ./my_binary --flag
# or: dotnet run
```

Then run:
```bash
harness run --agent ./my-agent --benchmark gaia-level1
```

### Virtual Environment Auto-Detection

When running a **directory-based Python agent**, the harness automatically checks for a virtual environment inside the agent directory. If `.venv/bin/python` or `venv/bin/python` exists, it will be used instead of the system `python`.

This is critical for agents that depend on packages not installed in the harness environment (e.g. `smolagents`, `pymupdf`, etc.).

```bash
# Example: agent with its own venv
agents/hal_generalist/
├── agent.py
├── requirements.txt
└── .venv/              # ← harness will use .venv/bin/python automatically
    └── bin/python

# If the venv lives elsewhere, symlink it:
ln -s ../../.venv-hal agents/hal_generalist/.venv

# Now the harness runs: .venv/bin/python agents/hal_generalist/agent.py
# instead of:          python agents/hal_generalist/agent.py
```

> **Important**: Without this, agents that import packages only available in their venv will fail with `ModuleNotFoundError` at runtime. If your agent has custom dependencies, always ensure a `.venv` exists in the agent directory (even as a symlink).

## Benchmarks

### Available Benchmarks

- `arithmetic` - Simple arithmetic problems (built-in, no dependencies)
- `gaia` - GAIA benchmark (requires `datasets` package)
- `gaia-level1`, `gaia-level2`, `gaia-level3` - GAIA by difficulty level
- `terminal-bench` - Terminal-Bench tasks (requires `terminal-bench` + Docker)
- `terminal-bench-core` - Terminal-Bench core dataset

```bash
# List available benchmarks
harness benchmarks

# Run GAIA Level 1
pip install datasets  # If not installed
harness run --agent ./my_agent --benchmark gaia-level1 --output ./results
```

### Terminal-Bench

[Terminal-Bench](https://github.com/terminal-bench/terminal-bench) evaluates agents on real-world terminal/DevOps tasks inside Docker containers. Unlike GAIA (where grading compares a string answer), Terminal-Bench grades by running a **test suite inside the container** after the agent finishes.

Each task provides:
- An **instruction** (what to accomplish)
- A **Docker environment** (`docker-compose.yaml`)
- A **test suite** (`run-tests.sh` + `tests/`) that checks the final container state

**Requirements**: `pip install terminal-bench` + a running Docker daemon.

```bash
# Install terminal-bench
pip install terminal-bench

# Run 3 easy tasks with the built-in terminal agent
harness run \
    --agent agents/terminal_agent.py \
    --benchmark terminal-bench \
    --dataset-name terminal-bench-core \
    --dataset-version 0.1.1 \
    --difficulty easy \
    --num-tasks 3 \
    --model openrouter/deepseek/deepseek-chat-v3-0324 \
    --task-timeout 600 \
    --parallel 1

# Run from a local dataset directory
harness run \
    --agent agents/terminal_agent.py \
    --benchmark terminal-bench \
    --dataset-path /path/to/local/tasks \
    --model openrouter/deepseek/deepseek-chat-v3-0324

# Run the full core dataset
harness run \
    --agent agents/terminal_agent.py \
    --benchmark terminal-bench-core \
    --model openrouter/deepseek/deepseek-chat-v3-0324 \
    --task-timeout 3600 \
    --parallel 2
```

#### Terminal-Bench CLI Options

| Option | Description |
|--------|-------------|
| `--dataset-name` | Dataset name in the TB registry (e.g. `terminal-bench-core`) |
| `--dataset-version` | Dataset version tag (e.g. `0.1.1`) |
| `--dataset-path` | Local path to a dataset directory (overrides name/version) |
| `--difficulty` | Filter tasks by difficulty: `easy`, `medium`, `hard` |

#### Terminal Agent

The built-in `terminal_agent.py` drives an LLM-in-the-loop shell interaction:

1. Starts a Docker container for each task
2. Captures an initial environment snapshot (`pwd`, `ls -la`)
3. Runs an LLM loop where the model issues one command at a time
4. Runs the task's test suite inside the container
5. Returns `PASS`/`FAIL` as the submission

Configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TB_MAX_ITERATIONS` | `30` | Maximum command iterations per task |
| `TB_COMMAND_TIMEOUT` | `120` | Per-command timeout in seconds |
| `TB_TEST_TIMEOUT` | `120` | Test suite timeout in seconds |

#### How Container Grading Works

Unlike string-comparison benchmarks, Terminal-Bench grading is **state-based**:

1. The agent interacts with a Docker container via shell commands
2. When done, the harness copies `run-tests.sh` + `tests/` into the container
3. The test suite runs inside the container and checks the final state
4. The pytest output is parsed to determine pass/fail

This means the agent's "submission" is the container state itself — the `PASS`/`FAIL` string is just a signal for the harness grading pipeline.

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

## Run Metadata

Every run produces a `run.json` with comprehensive metadata for analysis and database storage:

```json
{
  "run_id": "fb0df848",
  "timestamp": "2026-02-04T22:38:44.683841Z",
  "agent": "agents/metrics_agent.py",
  "benchmark": "arithmetic",
  "model": "openrouter/deepseek/deepseek-chat-v3-0324",
  "grader": "default",
  
  "git_commit": "64145b486f17",
  "git_branch": "main",
  "git_dirty": true,
  "run_command": "harness run --agent ... -b arithmetic ...",
  
  "num_tasks_run": 3,
  "num_tasks_success": 3,
  "num_tasks_failed": 0,
  "successful_task_ids": ["arith_000", "arith_001", "arith_002"],
  "failed_task_ids": [],
  
  "score": 100.0,
  "passed": 3,
  "total_graded": 3,
  
  "total_usage": {
    "prompt_tokens": 88,
    "completion_tokens": 5,
    "total_tokens": 93,
    "cached_tokens": 0,
    "reasoning_tokens": 0
  },
  "total_cost_usd": 0.00001372,
  "total_latency_ms": 10008.0,
  
  "model_stats": { ... },
  "task_stats": [ ... ],
  "agent_metrics": { ... }
}
```

### Token Tracking

The harness captures extended token usage across providers:

| Field | Description |
|-------|-------------|
| `prompt_tokens` | Input tokens |
| `completion_tokens` | Output tokens |
| `cached_tokens` | Cached input tokens (OpenAI, Anthropic) |
| `cache_creation_tokens` | Cache write tokens (Anthropic) |
| `reasoning_tokens` | Reasoning tokens (o1/o3, DeepSeek R1) |
| `audio_tokens` | Audio I/O tokens (OpenAI) |

## Agent Metrics

Agents can report custom KPIs that get aggregated across all tasks.

### In Your Agent

```python
class MyAgent(Agent):
    def run_task(self, task_id: str, task_data: dict) -> str:
        # Counter metrics
        self.increment("steps")
        self.increment("tokens_used", 150)
        
        # Arbitrary values
        self.metric("confidence", 0.95)
        self.metric("sources", ["web", "memory"])
        
        # Tool tracking (counts + sequence)
        self.record_tool_use("web_search", query="weather NYC")
        self.record_tool_use("calculator")
        
        return answer
```

### Aggregated Output

Metrics are aggregated in `run.json`:

```json
"agent_metrics": {
  "steps_total": 15,
  "steps_avg": 5.0,
  "steps_count": 3,
  "tool_sequence_all": ["web_search", "calculator", "llm_call", ...],
  "tool_counts_totals": {"web_search": 3, "calculator": 5, "llm_call": 10}
}
```

### Aggregation Rules

| Type | Aggregation |
|------|-------------|
| Numeric | `{name}_total`, `{name}_avg`, `{name}_count` |
| List | `{name}_all` (concatenated) |
| Dict (counters) | `{name}_totals` (summed per key) |
| Other | `{name}_values` (unique values) |

## Output Organization

Results are organized by benchmark and run ID:

```
results/
├── arithmetic/
│   ├── arithmetic_echo-agent_20260204_225130_5d8519/
│   │   ├── run.json        # Run metadata + aggregated stats
│   │   ├── summary.json    # Grading summary
│   │   ├── grades.json     # Per-task grades
│   │   ├── trace_arith_000.jsonl
│   │   └── trace_arith_001.jsonl
│   └── arithmetic_qa-agent_gpt-4o_20260204_230000_a1b2c3/
│       └── ...
├── gaia-level1/
│   └── gaia-level1_my-agent_claude-sonnet_20260205_140000_d4e5f6/
│       └── ...
└── custom/                 # For --tasks-file runs without --benchmark
    └── ...
```

### Run ID Format

Auto-generated run IDs include context for easy identification:

```
{benchmark}_{agent}_{model}_{YYYYMMDD_HHMMSS}_{random6}
```

Examples:
- `arithmetic_echo-agent_20260204_225130_5d8519` (no model)
- `gaia-level1_qa-agent_deepseek-chat-v3_20260204_230000_a1b2c3` (with model)

This provides:
- **Human readable**: Know what ran at a glance
- **Chronologically sortable**: Timestamp-based ordering
- **Collision resistant**: Timestamp + 24-bit random = billions of runs without collision

Options:
- **`--output`**: Base directory (default: `./results`)
- **`--run-id`**: Override with custom run ID

## Continuing Runs

The `harness continue` command re-runs tasks that failed or never completed. It works with:

- **Completed runs** (has `run.json`) — re-runs errored tasks
- **Interrupted runs** (has `status.jsonl` / `run_config.json`) — re-runs errored + incomplete tasks
- **Old interrupted runs** (only trace files) — scans traces for completion status, requires `--agent` and `--benchmark`

```bash
# Continue by run ID (exact or partial match)
harness continue 5d8519

# Continue an old interrupted run that has no config files
harness continue b5c291 \
    --agent agents/hal_generalist \
    --benchmark gaia \
    --model openrouter/deepseek/deepseek-chat-v3-0324 \
    --parallel 50 \
    --task-timeout 1800

# Continue by direct path
harness continue ./results/gaia/gaia_hal-generalist_*_b5c291/
```

### How Recovery Works

The harness recovers run state from whatever files are available, in priority order:

| Source | What it provides |
|--------|------------------|
| `run.json` | Full metadata from a completed run |
| `run_config.json` | Agent, benchmark, model, task IDs (written at run start) |
| `status.jsonl` | Real-time task results (append-only, crash-safe) |
| `trace_*.jsonl` | Scanned for `task_complete` / `task_error` events |
| CLI flags | `--agent`, `--benchmark`, `--model` override or supply missing config |

### Crash-Safe Progress Tracking

Every run now writes two recovery files:

- **`run_config.json`** — Written at the start of the run with full configuration and the list of all task IDs. This ensures the harness knows what was supposed to run even if the process is killed.
- **`status.jsonl`** — Append-only JSONL file written after each task completes or fails. Each line contains `task_id`, `status`, `submission`/`error`, `attempts`, `duration_ms`, and `timestamp`. Uses `flush()` for crash safety.

These files enable `harness continue` to pick up exactly where a killed run left off — no work is lost.

### Trace Scanning (Legacy Runs)

For old runs that pre-date `status.jsonl` (only have `trace_*.jsonl` files), the harness scans each trace for completion events:

| Trace content | Classification |
|---------------|----------------|
| Has `task_complete` event | **Completed** — submission preserved, won't re-run |
| Has `task_error` event | **Errored** — will be retried |
| Has `task_start` but no completion | **Interrupted** — will be retried |
| Empty file | **Incomplete** — will be retried |
| No trace file at all | **Never started** — discovered from benchmark, will be run |

When `--benchmark` is provided but no `run_config.json` exists, the harness discovers all benchmark tasks and marks any without traces as incomplete.

## CLI Reference

```bash
# Run single task
harness run-one --agent ./agent.py --task '{"id": "t1", "data": {"question": "2+2?"}}'

# Run benchmark (output defaults to ./results/{benchmark}/{run_id}/)
harness run --agent ./agent.py --benchmark gaia

# Run with custom output location and run ID
harness run \
    --agent ./agent.py \
    --benchmark gaia-level1 \
    --output ./my-results \
    --run-id experiment-001

# Run with all options
harness run \
    --agent ./agent.py \
    --benchmark gaia-level1 \
    --output ./results \
    --run-id my-run \
    --parallel 10 \
    --max-retries 3 \
    --task-timeout 300 \
    --num-tasks 50 \
    --model gpt-4o \
    --grader llm-fallback

# Run Terminal-Bench easy tasks
harness run \
    --agent agents/terminal_agent.py \
    --benchmark terminal-bench \
    --dataset-name terminal-bench-core \
    --dataset-version 0.1.1 \
    --difficulty easy \
    --model openrouter/deepseek/deepseek-chat-v3-0324 \
    --task-timeout 600 \
    --parallel 1

# List benchmarks
harness benchmarks

# View results
harness view ./results/gaia-level1/my-run

# Continue a failed or interrupted run
harness continue <run_id>
harness continue <run_id> --agent ./agent --benchmark gaia --parallel 50
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
    --agent agents/echo_agent.py \
    --task '{"id": "test", "data": {"x": 1}}'
```

## Roadmap

- [x] M1: Agent protocol + single task runner
- [x] M2: JSONL logging with raw API capture
- [x] M3: LiteLLM provider with auto-logging
- [x] M4: Parallel runner with retry logic
- [x] M5: Benchmark system with GAIA
- [x] M6: Run metadata with token/cost tracking
- [x] M7: Agent metrics system
- [x] LLM-as-judge grading layer

### Next Up

- [x] **Continue run**: `harness continue <run_id>` - Re-run errored/interrupted tasks with crash-safe recovery (`status.jsonl`, trace scanning, CLI overrides)
- [x] **HuggingFace integration**: Create HF dataset repo to store `run.json` files
- [x] **Push to HF**: `harness push <run_id>` - Upload run.json to HuggingFace dataset
- [x] **HAL Generalist Agent**: Port the [HAL Generalist Agent](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) GAIA scaffold and run full DeepSeek evaluation
- [x] **Terminal-Bench integration**: Container-graded terminal tasks with Docker+tmux, LLM-in-the-loop terminal agent
- [ ] **Full Terminal-Bench run**: DeepSeek V3 (`deepseek-chat-v3-0324`) on the full `terminal-bench-core` dataset via OpenRouter

### Future

- [ ] M8: Sandbox tiers (venv, firejail, docker)
- [ ] M9: Better viewer / dashboard
- [ ] More benchmarks (SWE-bench, GPQA, etc.)

## License

MIT

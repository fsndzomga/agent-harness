"""Tests for P0: Unified Benchmark Architecture v2 base refactor.

Covers:
  - ExecutionMode enum and all 6 modes
  - Task dataclass (v2 fields: execution_mode, environment, metadata)
  - TaskEnvironment dataclass
  - ExecutionContext dataclass
  - Benchmark ABC contract (grade signature, setup/teardown)
  - Agent ABC (AgentBase) with mode-specific methods and supports_mode()
  - TaskOrchestrator DIRECT mode dispatch + error handling
  - GradingPipeline layered matchers
  - Migrated benchmarks: Arithmetic, GAIA (lightweight checks)
  - New DIRECT benchmark stubs: AssistantBench, HLE, ARC-AGI, BrowseComp
  - Benchmark registry: new benchmarks discoverable
  - Backward compatibility: grade_legacy, grade_all, Task(id, data) still works
"""

import json
import pytest

from harness.protocol import Task

from harness.benchmarks.base import (
    ExecutionMode,
    TaskEnvironment,
    ExecutionContext,
    GradeResult,
    Benchmark,
)
from harness.benchmarks import (
    get_benchmark,
    list_benchmarks,
)
from harness.agents.base import AgentBase
from harness.execution.orchestrator import TaskOrchestrator
from harness.grading.pipeline import GradingPipeline


# ============================================================================
# ExecutionMode
# ============================================================================

class TestExecutionMode:
    """Verify all 6 execution archetypes are defined."""

    def test_all_modes_exist(self):
        modes = [m.value for m in ExecutionMode]
        assert "direct" in modes
        assert "code_submit" in modes
        assert "interactive" in modes
        assert "conversational" in modes
        assert "tool_use" in modes
        assert "gui_agent" in modes

    def test_mode_count(self):
        assert len(ExecutionMode) == 6

    def test_mode_values_are_strings(self):
        for mode in ExecutionMode:
            assert isinstance(mode.value, str)


# ============================================================================
# Task dataclass (v2 fields)
# ============================================================================

class TestTaskV2:
    """Task dataclass backward compat + new fields."""

    def test_basic_creation(self):
        """Old-style Task(id, data) still works."""
        t = Task(id="t1", data={"question": "hi"})
        assert t.id == "t1"
        assert t.data == {"question": "hi"}
        assert t.execution_mode is None
        assert t.environment is None
        assert t.metadata == {}

    def test_v2_fields(self):
        env = TaskEnvironment(docker_image="python:3.12")
        t = Task(
            id="t2",
            data={"q": "test"},
            execution_mode="direct",
            environment=env,
            metadata={"level": 1},
        )
        assert t.execution_mode == "direct"
        assert t.environment.docker_image == "python:3.12"
        assert t.metadata["level"] == 1

    def test_to_jsonrpc(self):
        """JSON-RPC format remains unchanged."""
        t = Task(id="t1", data={"question": "hi"}, execution_mode="direct")
        msg = json.loads(t.to_jsonrpc())
        assert msg["method"] == "run_task"
        assert msg["params"]["task_id"] == "t1"

    def test_from_dict(self):
        d = {"id": "t3", "data": {"q": "x"}, "execution_mode": "interactive"}
        t = Task.from_dict(d)
        assert t.id == "t3"
        assert t.execution_mode == "interactive"


# ============================================================================
# TaskEnvironment
# ============================================================================

class TestTaskEnvironment:

    def test_defaults(self):
        env = TaskEnvironment()
        assert env.docker_image is None
        assert env.docker_compose is None
        assert env.vm_image is None
        assert env.vm_platform is None
        assert env.tool_servers == []
        assert env.setup_script is None
        assert env.teardown_script is None
        assert env.eval_script is None

    def test_docker_env(self):
        env = TaskEnvironment(docker_image="ubuntu:22.04", setup_script="init.sh")
        assert env.docker_image == "ubuntu:22.04"
        assert env.setup_script == "init.sh"

    def test_tool_use_env(self):
        env = TaskEnvironment(
            docker_compose="compose.yaml",
            tool_servers=["exa", "brave", "postgres"],
        )
        assert len(env.tool_servers) == 3

    def test_gui_env(self):
        env = TaskEnvironment(
            vm_image="osworld-ubuntu.qcow2",
            vm_platform="docker",
            eval_script="eval_task_123.py",
        )
        assert env.vm_platform == "docker"


# ============================================================================
# ExecutionContext
# ============================================================================

class TestExecutionContext:

    def test_defaults(self):
        ctx = ExecutionContext(task_id="t1")
        assert ctx.task_id == "t1"
        assert ctx.container_id is None
        assert ctx.vm_id is None
        assert ctx.tool_server_ids == []
        assert ctx.env_state == {}
        assert ctx.history == []
        assert ctx.action_count == 0
        assert ctx.cost_usd == 0.0
        assert ctx.model_costs == {}
        assert ctx.model_tokens == {}

    def test_cost_tracking(self):
        ctx = ExecutionContext(task_id="t1")
        ctx.cost_usd = 0.05
        ctx.model_costs["claude"] = 0.03
        ctx.model_costs["gpt-4o"] = 0.02
        ctx.model_tokens["claude"] = {"input": 12000, "output": 3400}
        assert ctx.cost_usd == pytest.approx(0.05)
        assert ctx.model_tokens["claude"]["input"] == 12000


# ============================================================================
# Benchmark ABC contract
# ============================================================================

class TestBenchmarkABC:

    def test_cannot_instantiate_bare(self):
        with pytest.raises(TypeError):
            Benchmark()  # type: ignore[abstract]

    def test_must_implement_get_tasks(self):
        class Partial(Benchmark):
            def grade(self, task, result, context):
                pass
        with pytest.raises(TypeError):
            Partial()

    def test_must_implement_grade(self):
        class Partial(Benchmark):
            def get_tasks(self):
                return []
        with pytest.raises(TypeError):
            Partial()

    def test_complete_subclass(self):
        class MyBench(Benchmark):
            name = "test"
            execution_mode = ExecutionMode.DIRECT

            def get_tasks(self):
                return [Task(id="t1", data={"q": "hi"})]

            def grade(self, task, result, context):
                return GradeResult(
                    task_id=task.id,
                    passed=result == "hello",
                    score=1.0 if result == "hello" else 0.0,
                    expected="hello",
                    actual=str(result),
                    method="exact",
                )

        b = MyBench()
        assert b.name == "test"
        assert b.execution_mode == ExecutionMode.DIRECT
        tasks = b.get_tasks()
        assert len(tasks) == 1

        ctx = ExecutionContext(task_id="t1")
        grade = b.grade(tasks[0], "hello", ctx)
        assert isinstance(grade, GradeResult)
        assert grade.passed is True

    def test_setup_teardown_defaults(self):
        class MyBench(Benchmark):
            name = "test"
            def get_tasks(self):
                return [Task(id="t1", data={})]
            def grade(self, task, result, context):
                return GradeResult(
                    task_id=task.id, passed=True, score=1.0,
                    expected="", actual="", method="test",
                )

        b = MyBench()
        task = b.get_tasks()[0]
        ctx = b.setup_environment(task)
        assert isinstance(ctx, ExecutionContext)
        assert ctx.task_id == "t1"
        b.teardown_environment(ctx)  # Should not raise

    def test_grade_legacy(self):
        """grade_legacy bridges old (task_id, submission) to new (task, result, ctx)."""
        class MyBench(Benchmark):
            name = "test"
            def get_tasks(self):
                return [Task(id="t1", data={"q": "hi"})]
            def grade(self, task, result, context):
                return GradeResult(
                    task_id=task.id,
                    passed=result == "42",
                    score=1.0 if result == "42" else 0.0,
                    expected="42",
                    actual=str(result),
                    method="exact",
                )

        b = MyBench()
        grade = b.grade_legacy("t1", "42")
        assert grade.passed is True
        assert grade.task_id == "t1"

    def test_grade_legacy_missing_task(self):
        class MyBench(Benchmark):
            name = "test"
            def get_tasks(self):
                return []
            def grade(self, task, result, context):
                pass

        b = MyBench()
        with pytest.raises(ValueError, match="not found"):
            b.grade_legacy("missing", "answer")

    def test_grade_all(self):
        class MyBench(Benchmark):
            name = "test"
            def get_tasks(self):
                return [
                    Task(id="t1", data={}),
                    Task(id="t2", data={}),
                ]
            def grade(self, task, result, context):
                return GradeResult(
                    task_id=task.id,
                    passed=result == "ok",
                    score=1.0 if result == "ok" else 0.0,
                    expected="ok",
                    actual=str(result),
                    method="exact",
                )

        b = MyBench()
        results = [
            {"task_id": "t1", "submission": "ok"},
            {"task_id": "t2", "submission": "nope"},
        ]
        grades = b.grade_all(results)
        assert len(grades) == 2
        assert grades[0].passed is True
        assert grades[1].passed is False


# ============================================================================
# AgentBase
# ============================================================================

class TestAgentBase:

    def test_cannot_instantiate_directly(self):
        """AgentBase is ABC — but has no abstract methods, so it CAN be instantiated.
        All methods raise NotImplementedError instead."""
        agent = type("Dummy", (AgentBase,), {})()
        with pytest.raises(NotImplementedError, match="DIRECT"):
            agent.run_task("t1", {})

    def test_all_modes_raise_not_implemented(self):
        agent = type("Dummy", (AgentBase,), {})()
        with pytest.raises(NotImplementedError, match="DIRECT"):
            agent.run_task("t1", {})
        with pytest.raises(NotImplementedError, match="CODE_SUBMIT"):
            agent.run_code_task("t1", {})
        with pytest.raises(NotImplementedError, match="INTERACTIVE"):
            agent.run_interactive_task("t1", {}, lambda cmd: "")
        with pytest.raises(NotImplementedError, match="CONVERSATIONAL"):
            agent.run_conversational_task("t1", {}, lambda msg: "", lambda n, p: None)
        with pytest.raises(NotImplementedError, match="TOOL_USE"):
            agent.run_tool_task("t1", {}, lambda s, t, p: None)
        with pytest.raises(NotImplementedError, match="GUI_AGENT"):
            agent.run_gui_task("t1", {}, lambda: {}, lambda a: {})

    def test_supports_mode_false_for_base(self):
        agent = type("Dummy", (AgentBase,), {})()
        assert agent.supports_mode("direct") is False
        assert agent.supports_mode("interactive") is False

    def test_supports_mode_true_when_overridden(self):
        class DirectAgent(AgentBase):
            def run_task(self, task_id, task_data):
                return "answer"

        agent = DirectAgent()
        assert agent.supports_mode("direct") is True
        assert agent.supports_mode("interactive") is False
        assert agent.supports_mode("code_submit") is False

    def test_supports_mode_unknown(self):
        agent = type("Dummy", (AgentBase,), {})()
        assert agent.supports_mode("unknown_mode") is False

    def test_direct_agent_works(self):
        class EchoAgent(AgentBase):
            def run_task(self, task_id, task_data):
                return task_data.get("question", "")

        agent = EchoAgent()
        result = agent.run_task("t1", {"question": "hello"})
        assert result == "hello"


# ============================================================================
# TaskOrchestrator
# ============================================================================

class TestTaskOrchestrator:

    def _make_bench(self, mode=ExecutionMode.DIRECT, answers=None):
        """Create a minimal benchmark for testing."""
        answers = answers or {"t1": "42", "t2": "Paris"}

        class TestBench(Benchmark):
            name = "test"
            execution_mode = mode

            def get_tasks(self):
                return [
                    Task(id=tid, data={"question": f"Q for {tid}"})
                    for tid in answers
                ]

            def grade(self_, task, result, context):
                expected = answers.get(task.id, "")
                passed = str(result).strip() == expected
                return GradeResult(
                    task_id=task.id,
                    passed=passed,
                    score=1.0 if passed else 0.0,
                    expected=expected,
                    actual=str(result),
                    method="exact" if passed else "none",
                )

        return TestBench()

    def _make_agent(self, answer_map=None):
        answer_map = answer_map or {}

        class TestAgent(AgentBase):
            def run_task(self, task_id, task_data):
                return answer_map.get(task_id, "")

        return TestAgent()

    def test_direct_dispatch_pass(self):
        bench = self._make_bench(answers={"t1": "42"})
        agent = self._make_agent({"t1": "42"})
        orch = TaskOrchestrator()

        task = bench.get_tasks()[0]
        grade = orch.run(task, agent, bench)
        assert grade.passed is True
        assert grade.method == "exact"

    def test_direct_dispatch_fail(self):
        bench = self._make_bench(answers={"t1": "42"})
        agent = self._make_agent({"t1": "wrong"})
        orch = TaskOrchestrator()

        task = bench.get_tasks()[0]
        grade = orch.run(task, agent, bench)
        assert grade.passed is False

    def test_run_all(self):
        bench = self._make_bench(answers={"t1": "42", "t2": "Paris"})
        agent = self._make_agent({"t1": "42", "t2": "Paris"})
        orch = TaskOrchestrator()

        grades = orch.run_all(bench, agent)
        assert len(grades) == 2
        assert all(g.passed for g in grades)

    def test_run_all_subset(self):
        bench = self._make_bench(answers={"t1": "42", "t2": "Paris"})
        agent = self._make_agent({"t1": "42"})
        orch = TaskOrchestrator()

        grades = orch.run_all(bench, agent, task_ids=["t1"])
        assert len(grades) == 1
        assert grades[0].task_id == "t1"

    def test_run_all_handles_errors(self):
        """If agent throws, the task gets a failing grade."""
        bench = self._make_bench(answers={"t1": "42"})

        class FailAgent(AgentBase):
            def run_task(self, task_id, task_data):
                raise RuntimeError("Agent crashed")

        orch = TaskOrchestrator()
        grades = orch.run_all(bench, FailAgent())
        assert len(grades) == 1
        assert grades[0].passed is False
        assert "Agent crashed" in grades[0].actual

    def test_non_direct_modes_raise(self):
        for mode in [
            ExecutionMode.CODE_SUBMIT,
            ExecutionMode.INTERACTIVE,
            ExecutionMode.CONVERSATIONAL,
            ExecutionMode.TOOL_USE,
            ExecutionMode.GUI_AGENT,
        ]:
            bench = self._make_bench(mode=mode, answers={"t1": "x"})
            agent = self._make_agent({"t1": "x"})
            orch = TaskOrchestrator()
            task = bench.get_tasks()[0]
            with pytest.raises(NotImplementedError):
                orch.run(task, agent, bench)

    def test_teardown_called_on_success(self):
        """Verify teardown is called even on success."""
        teardown_called = []

        class TeardownBench(Benchmark):
            name = "td"
            execution_mode = ExecutionMode.DIRECT

            def get_tasks(self):
                return [Task(id="t1", data={})]

            def grade(self, task, result, context):
                return GradeResult(
                    task_id=task.id, passed=True, score=1.0,
                    expected="", actual="", method="test",
                )

            def teardown_environment(self, context):
                teardown_called.append(context.task_id)

        agent = self._make_agent({"t1": "ok"})
        orch = TaskOrchestrator()
        orch.run(TeardownBench().get_tasks()[0], agent, TeardownBench())
        assert "t1" in teardown_called

    def test_teardown_called_on_failure(self):
        """Verify teardown is called even if agent raises."""
        teardown_called = []

        class TeardownBench(Benchmark):
            name = "td"
            execution_mode = ExecutionMode.DIRECT

            def get_tasks(self):
                return [Task(id="t1", data={})]

            def grade(self, task, result, context):
                return GradeResult(
                    task_id=task.id, passed=False, score=0.0,
                    expected="", actual="", method="error",
                )

            def teardown_environment(self, context):
                teardown_called.append(context.task_id)

        class FailAgent(AgentBase):
            def run_task(self, task_id, task_data):
                raise RuntimeError("boom")

        orch = TaskOrchestrator()
        with pytest.raises(RuntimeError):
            orch.run(TeardownBench().get_tasks()[0], FailAgent(), TeardownBench())
        assert "t1" in teardown_called


# ============================================================================
# GradingPipeline
# ============================================================================

class TestGradingPipeline:

    def test_default_matchers(self):
        p = GradingPipeline()
        assert p.matcher_names == ["exact", "normalized", "numeric", "fuzzy"]

    def test_exact_match(self):
        p = GradingPipeline()
        passed, method = p.match("42", "42")
        assert passed is True
        assert method == "exact"

    def test_normalized_match(self):
        p = GradingPipeline()
        passed, method = p.match("Hello World", "hello world")
        assert passed is True
        assert method == "normalized"

    def test_numeric_match(self):
        p = GradingPipeline()
        passed, method = p.match("The answer is 42", "42")
        assert passed is True
        assert method == "numeric"

    def test_fuzzy_match(self):
        p = GradingPipeline()
        passed, method = p.match("hello world!", "hello world")
        assert passed is True
        # Could be exact or fuzzy depending on normalization
        assert method in ("exact", "normalized", "fuzzy")

    def test_no_match(self):
        p = GradingPipeline()
        passed, method = p.match("completely wrong", "correct answer")
        assert passed is False
        assert method == "none"

    def test_custom_matchers(self):
        p = GradingPipeline(matchers=["exact"])
        assert p.matcher_names == ["exact"]
        # Normalized won't work with only exact
        passed, _ = p.match("Hello", "hello")
        assert passed is False

    def test_contains_matcher(self):
        p = GradingPipeline(matchers=["contains"])
        passed, method = p.match("The answer is 42 degrees", "42")
        assert passed is True
        assert method == "contains"

    def test_grade_answer(self):
        p = GradingPipeline()
        grade = p.grade_answer("42", "42", task_id="t1")
        assert isinstance(grade, GradeResult)
        assert grade.passed is True
        assert grade.task_id == "t1"
        assert grade.method == "exact"

    def test_invalid_matcher_raises(self):
        with pytest.raises(ValueError, match="Unknown matcher"):
            GradingPipeline(matchers=["nonexistent"])

    def test_register_custom_matcher(self):
        GradingPipeline.register_matcher(
            "always_true", lambda s, e: True
        )
        assert "always_true" in GradingPipeline.available_matchers()
        p = GradingPipeline(matchers=["always_true"])
        passed, method = p.match("anything", "everything")
        assert passed is True
        assert method == "always_true"

    def test_available_matchers(self):
        names = GradingPipeline.available_matchers()
        assert "exact" in names
        assert "normalized" in names
        assert "numeric" in names
        assert "fuzzy" in names
        assert "contains" in names


# ============================================================================
# Migrated benchmarks
# ============================================================================

class TestArithmeticMigrated:

    def test_arithmetic_execution_mode(self):
        bench = get_benchmark("arithmetic")
        assert bench.execution_mode == ExecutionMode.DIRECT

    def test_arithmetic_grade_new_signature(self):
        bench = get_benchmark("arithmetic", num_tasks=5)
        tasks = bench.get_tasks()
        assert len(tasks) == 5

        t = tasks[0]
        ctx = ExecutionContext(task_id=t.id)

        # Compute the correct answer
        expected = bench._answers[t.id]
        grade = bench.grade(t, expected, ctx)
        assert isinstance(grade, GradeResult)
        assert grade.passed is True

    def test_arithmetic_grade_wrong(self):
        bench = get_benchmark("arithmetic", num_tasks=1)
        t = bench.get_tasks()[0]
        ctx = ExecutionContext(task_id=t.id)
        grade = bench.grade(t, "wrong answer", ctx)
        assert grade.passed is False

    def test_arithmetic_grade_legacy(self):
        """Backward compat: grade_legacy still works."""
        bench = get_benchmark("arithmetic", num_tasks=1)
        t = bench.get_tasks()[0]
        expected = bench._answers[t.id]
        grade = bench.grade_legacy(t.id, expected)
        assert grade.passed is True

    def test_arithmetic_with_orchestrator(self):
        """End-to-end: orchestrator + arithmetic benchmark."""
        bench = get_benchmark("arithmetic", num_tasks=3)

        class ArithAgent(AgentBase):
            def __init__(self, answers):
                self._answers = answers
            def run_task(self, task_id, task_data):
                return self._answers.get(task_id, "")

        agent = ArithAgent(bench._answers)
        orch = TaskOrchestrator()
        grades = orch.run_all(bench, agent)
        assert len(grades) == 3
        assert all(g.passed for g in grades)


# ============================================================================
# New DIRECT benchmark stubs
# ============================================================================

class TestNewDirectBenchmarks:

    @pytest.mark.parametrize("name", [
        "assistant-bench",
        "hle",
        "arc-agi",
        "arc-agi-1",
        "arc-agi-2",
        "browsecomp",
    ])
    def test_benchmark_in_registry(self, name):
        bench = get_benchmark(name)
        assert bench.name == name
        assert bench.execution_mode == ExecutionMode.DIRECT

    @pytest.mark.parametrize("name", [
        "assistant-bench",
        "hle",
        "arc-agi",
        "arc-agi-1",
        "arc-agi-2",
        "browsecomp",
    ])
    def test_get_tasks_returns_list(self, name):
        bench = get_benchmark(name)
        try:
            tasks = bench.get_tasks()
        except ImportError as exc:
            pytest.skip(f"Missing dependency: {exc}")
        assert isinstance(tasks, list)

    def test_all_new_benchmarks_in_list(self):
        names = list_benchmarks()
        for expected in [
            "assistant-bench", "hle", "arc-agi", "arc-agi-1",
            "arc-agi-2", "browsecomp",
        ]:
            assert expected in names, f"{expected} not in registry"


class TestARCAGIGrading:
    """ARC-AGI has custom grid-match grading."""

    def test_exact_grid_match(self):
        from harness.benchmarks.arc_agi import ARCAGIBenchmark

        bench = ARCAGIBenchmark()
        bench._tasks = [Task(id="arc1", data={"question": "predict grid"})]
        bench._expected_outputs = {
            "arc1": [[[1, 2], [3, 4]]],  # one test input with expected output
        }

        task = bench.get_tasks()[0]
        ctx = ExecutionContext(task_id="arc1")

        # Correct prediction
        grade = bench.grade(task, [[[1, 2], [3, 4]]], ctx)
        assert grade.passed is True
        assert grade.method == "grid_match"

    def test_wrong_grid(self):
        from harness.benchmarks.arc_agi import ARCAGIBenchmark

        bench = ARCAGIBenchmark()
        bench._tasks = [Task(id="arc1", data={})]
        bench._expected_outputs = {"arc1": [[[1, 2], [3, 4]]]}

        task = bench.get_tasks()[0]
        ctx = ExecutionContext(task_id="arc1")
        grade = bench.grade(task, [[[0, 0], [0, 0]]], ctx)
        assert grade.passed is False

    def test_json_string_input(self):
        from harness.benchmarks.arc_agi import ARCAGIBenchmark

        bench = ARCAGIBenchmark()
        bench._tasks = [Task(id="arc1", data={})]
        bench._expected_outputs = {"arc1": [[[5, 6], [7, 8]]]}

        task = bench.get_tasks()[0]
        ctx = ExecutionContext(task_id="arc1")
        grade = bench.grade(task, json.dumps([[[5, 6], [7, 8]]]), ctx)
        assert grade.passed is True

    def test_multiple_attempts(self):
        from harness.benchmarks.arc_agi import ARCAGIBenchmark

        bench = ARCAGIBenchmark(max_attempts=3)
        bench._tasks = [Task(id="arc1", data={})]
        bench._expected_outputs = {"arc1": [[[1, 0], [0, 1]]]}

        task = bench.get_tasks()[0]
        ctx = ExecutionContext(task_id="arc1")

        # 3 attempts: first two wrong, third correct
        result = [[
            [[0, 0], [0, 0]],
            [[1, 1], [1, 1]],
            [[1, 0], [0, 1]],
        ]]
        grade = bench.grade(task, result, ctx)
        assert grade.passed is True


# ============================================================================
# Benchmark registry completeness
# ============================================================================

class TestBenchmarkRegistry:

    def test_all_original_benchmarks_still_available(self):
        names = list_benchmarks()
        assert "arithmetic" in names
        # GAIA requires `datasets` so may not be registered —
        # just check it doesn't crash
        # Terminal-Bench requires its SDK so may not be registered

    def test_new_benchmarks_registered(self):
        names = list_benchmarks()
        for name in [
            "assistant-bench", "hle", "browsecomp",
            "arc-agi", "arc-agi-1", "arc-agi-2",
        ]:
            assert name in names

    def test_get_benchmark_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_benchmark("does-not-exist")


# ============================================================================
# Integration: full DIRECT pipeline
# ============================================================================

class TestDirectPipelineIntegration:
    """End-to-end test: benchmark → orchestrator → agent → grading."""

    def test_full_pipeline(self):
        # Create a benchmark with known answers
        class QuizBench(Benchmark):
            name = "quiz"
            execution_mode = ExecutionMode.DIRECT

            def get_tasks(self):
                return [
                    Task(id="q1", data={"question": "Capital of France?"}),
                    Task(id="q2", data={"question": "2 + 2?"}),
                    Task(id="q3", data={"question": "Color of the sky?"}),
                ]

            def grade(self, task, result, context):
                answers = {"q1": "Paris", "q2": "4", "q3": "blue"}
                expected = answers[task.id]
                pipeline = GradingPipeline()
                return pipeline.grade_answer(
                    str(result), expected, task_id=task.id
                )

        # Create an agent that knows the answers
        class SmartAgent(AgentBase):
            def run_task(self, task_id, task_data):
                answers = {"q1": "paris", "q2": "4", "q3": "Blue"}
                return answers.get(task_id, "")

        bench = QuizBench()
        agent = SmartAgent()
        orch = TaskOrchestrator()

        grades = orch.run_all(bench, agent)
        assert len(grades) == 3
        # "paris" vs "Paris" → normalized match
        assert grades[0].passed is True
        # "4" vs "4" → exact match
        assert grades[1].passed is True
        # "Blue" vs "blue" → normalized match
        assert grades[2].passed is True

        # Score summary
        total = sum(1 for g in grades if g.passed)
        assert total == 3

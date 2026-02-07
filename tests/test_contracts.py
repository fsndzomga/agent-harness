"""Tests for the completion logging contract.

Validates:
- BaseProvider ABC enforces _do_complete()
- BaseProvider.complete() auto-logs completions
- Agent.emit_completion() emits proper JSON-RPC
- Agent warns on zero completions
- Standalone emit_completion() emits proper JSON-RPC
"""

import json
import sys
import io
import pytest

from harness.providers.base import (
    BaseProvider,
    Provider,
    Message,
    CompletionRequest,
    CompletionResponse,
    Usage,
)
from harness.agent import Agent
from harness.logging import StdoutLogger, emit_completion


# =====================================================================
# BaseProvider contract tests
# =====================================================================


class TestBaseProviderContract:
    """BaseProvider must enforce _do_complete() and auto-log."""

    def test_cannot_instantiate_without_do_complete(self):
        """Subclass that doesn't implement _do_complete() cannot be instantiated."""

        class BadProvider(BaseProvider):
            pass  # forgot to implement _do_complete

        with pytest.raises(TypeError, match="_do_complete"):
            BadProvider(model="test-model")

    def test_subclass_with_do_complete_works(self):
        """Subclass that implements _do_complete() can be instantiated."""

        class GoodProvider(BaseProvider):
            def _do_complete(self, request):
                return CompletionResponse(
                    message=Message(role="assistant", content="hi"),
                    usage=Usage(input_tokens=10, output_tokens=5),
                    raw_request={"model": "test"},
                    raw_response={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
                    provider="test",
                    latency_ms=42,
                )

        provider = GoodProvider(model="test-model")
        assert provider.model == "test-model"

    def test_complete_calls_do_complete(self):
        """complete() delegates to _do_complete()."""

        class TestProvider(BaseProvider):
            called = False

            def _do_complete(self, request):
                TestProvider.called = True
                return CompletionResponse(
                    message=Message(role="assistant", content="answer"),
                    usage=Usage(input_tokens=10, output_tokens=5),
                    raw_request={"model": self.model},
                    raw_response={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
                    provider="test",
                    latency_ms=10,
                )

        provider = TestProvider(model="test-model")
        req = CompletionRequest(messages=[Message(role="user", content="hi")])
        result = provider.complete(req)

        assert TestProvider.called
        assert result.message.content == "answer"

    def test_complete_auto_logs(self):
        """complete() automatically calls logger.log_completion()."""
        logged = []

        class FakeLogger:
            def log_completion(self, provider, request, response, latency_ms):
                logged.append({
                    "provider": provider,
                    "request": request,
                    "response": response,
                    "latency_ms": latency_ms,
                })

        class TestProvider(BaseProvider):
            def _do_complete(self, request):
                return CompletionResponse(
                    message=Message(role="assistant", content="x"),
                    usage=Usage(input_tokens=100, output_tokens=50),
                    raw_request={"model": self.model, "messages": []},
                    raw_response={"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
                    provider="test-provider",
                    latency_ms=99,
                )

        provider = TestProvider(model="m", logger=FakeLogger())
        req = CompletionRequest(messages=[Message(role="user", content="hi")])
        provider.complete(req)

        assert len(logged) == 1
        assert logged[0]["provider"] == "test-provider"
        assert logged[0]["latency_ms"] == 99
        assert logged[0]["response"]["usage"]["prompt_tokens"] == 100

    def test_complete_no_logger_still_works(self):
        """complete() works fine without a logger (no crash)."""

        class TestProvider(BaseProvider):
            def _do_complete(self, request):
                return CompletionResponse(
                    message=Message(role="assistant", content="ok"),
                    usage=Usage(input_tokens=1, output_tokens=1),
                    raw_request={}, raw_response={},
                    provider="p", latency_ms=0,
                )

        provider = TestProvider(model="m")  # no logger
        req = CompletionRequest(messages=[Message(role="user", content="hi")])
        result = provider.complete(req)
        assert result.message.content == "ok"

    def test_satisfies_provider_protocol(self):
        """BaseProvider subclasses satisfy the Provider Protocol."""

        class TestProvider(BaseProvider):
            def _do_complete(self, request):
                return CompletionResponse(
                    message=Message(role="assistant", content="ok"),
                    usage=Usage(input_tokens=1, output_tokens=1),
                    raw_request={}, raw_response={},
                    provider="p", latency_ms=0,
                )

        provider = TestProvider(model="m")
        assert isinstance(provider, Provider)


# =====================================================================
# Agent.emit_completion() tests
# =====================================================================


class TestAgentEmitCompletion:
    """Agent.emit_completion() must emit proper completion events."""

    def test_emit_completion_writes_jsonrpc(self, capsys):
        """emit_completion() writes a valid JSON-RPC log message to stdout."""
        agent = Agent()

        agent.emit_completion(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            provider="openai",
        )

        captured = capsys.readouterr()
        msg = json.loads(captured.out.strip())

        assert msg["jsonrpc"] == "2.0"
        assert msg["method"] == "log"
        assert msg["params"]["type"] == "completion"
        assert msg["params"]["provider"] == "openai"
        assert msg["params"]["response"]["usage"]["prompt_tokens"] == 100
        assert msg["params"]["response"]["usage"]["completion_tokens"] == 50
        assert msg["params"]["response"]["usage"]["total_tokens"] == 150

    def test_emit_completion_increments_count(self):
        """emit_completion() increments _completion_count."""
        agent = Agent()
        assert agent._completion_count == 0

        agent.emit_completion(model="m", input_tokens=1, output_tokens=1)
        assert agent._completion_count == 1

        agent.emit_completion(model="m", input_tokens=2, output_tokens=2)
        assert agent._completion_count == 2

    def test_emit_completion_with_cost(self, capsys):
        """emit_completion() includes cost in response when provided."""
        agent = Agent()
        agent.emit_completion(
            model="m", input_tokens=10, output_tokens=5, cost_usd=0.001,
        )

        captured = capsys.readouterr()
        msg = json.loads(captured.out.strip())
        assert msg["params"]["response"]["_cost_usd"] == 0.001

    def test_emit_completion_with_raw_response(self, capsys):
        """emit_completion() uses raw_response when provided."""
        agent = Agent()
        raw = {"model": "m", "usage": {"prompt_tokens": 99, "completion_tokens": 88, "total_tokens": 187}}
        agent.emit_completion(model="m", raw_response=raw)

        captured = capsys.readouterr()
        msg = json.loads(captured.out.strip())
        # Should use the raw_response's usage, not construct new one
        assert msg["params"]["response"]["usage"]["prompt_tokens"] == 99


# =====================================================================
# Zero-completion warning tests
# =====================================================================


class TestZeroCompletionWarning:
    """Agent must warn when run_task() emits no completions."""

    def test_warns_on_zero_completions(self, capsys, monkeypatch):
        """A warning is printed to stderr if run_task completes without completions."""

        class NoCompletionsAgent(Agent):
            def run_task(self, task_id, task_data):
                return "some answer"  # No self.complete() or self.emit_completion()

        agent = NoCompletionsAgent()

        # Simulate a single JSON-RPC task on stdin
        task_msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "run_task",
            "params": {"task_id": "t1", "task_data": {"question": "hi"}},
            "id": 1,
        })
        monkeypatch.setattr("sys.stdin", io.StringIO(task_msg + "\n"))

        agent.run()

        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "No completion events" in captured.err
        assert "emit_completion" in captured.err

    def test_no_warning_when_completions_emitted(self, capsys, monkeypatch):
        """No warning when emit_completion() is called during run_task()."""

        class GoodAgent(Agent):
            def run_task(self, task_id, task_data):
                self.emit_completion(model="m", input_tokens=10, output_tokens=5)
                return "answer"

        agent = GoodAgent()

        task_msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "run_task",
            "params": {"task_id": "t1", "task_data": {"question": "hi"}},
            "id": 1,
        })
        monkeypatch.setattr("sys.stdin", io.StringIO(task_msg + "\n"))

        agent.run()

        captured = capsys.readouterr()
        assert "WARNING" not in captured.err


# =====================================================================
# Standalone emit_completion() tests
# =====================================================================


class TestStandaloneEmitCompletion:
    """The standalone emit_completion() helper must emit proper JSON-RPC."""

    def test_emits_valid_jsonrpc(self, capsys):
        emit_completion(
            model="gpt-4o",
            input_tokens=200,
            output_tokens=100,
            provider="openai",
            latency_ms=500,
        )

        captured = capsys.readouterr()
        msg = json.loads(captured.out.strip())

        assert msg["jsonrpc"] == "2.0"
        assert msg["method"] == "log"
        params = msg["params"]
        assert params["type"] == "completion"
        assert params["provider"] == "openai"
        assert params["latency_ms"] == 500
        assert params["request"]["model"] == "gpt-4o"
        assert params["response"]["usage"]["prompt_tokens"] == 200
        assert params["response"]["usage"]["completion_tokens"] == 100
        assert params["response"]["usage"]["total_tokens"] == 300

    def test_includes_cost_when_provided(self, capsys):
        emit_completion(model="m", input_tokens=1, output_tokens=1, cost_usd=0.05)

        captured = capsys.readouterr()
        msg = json.loads(captured.out.strip())
        assert msg["params"]["response"]["_cost_usd"] == 0.05

    def test_no_cost_key_when_zero(self, capsys):
        emit_completion(model="m", input_tokens=1, output_tokens=1)

        captured = capsys.readouterr()
        msg = json.loads(captured.out.strip())
        assert "_cost_usd" not in msg["params"]["response"]

    def test_has_timestamp(self, capsys):
        emit_completion(model="m", input_tokens=1, output_tokens=1)

        captured = capsys.readouterr()
        msg = json.loads(captured.out.strip())
        assert "timestamp" in msg["params"]
        assert isinstance(msg["params"]["timestamp"], float)

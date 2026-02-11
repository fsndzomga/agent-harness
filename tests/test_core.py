"""Tests for agent harness core functionality."""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from harness.protocol import Task, Submission
from harness.runner import AgentRunner
from harness.logging import TraceLogger
from harness.benchmarks.graders import (
    exact_match,
    normalized_match,
    numeric_match,
    fuzzy_match,
    grade_with_pipeline,
)


class TestProtocol:
    """Test protocol message formatting."""
    
    def test_task_to_jsonrpc(self):
        task = Task(id="test_1", data={"question": "What is 2+2?"})
        msg = json.loads(task.to_jsonrpc())
        
        assert msg["jsonrpc"] == "2.0"
        assert msg["method"] == "run_task"
        assert msg["params"]["task_id"] == "test_1"
        assert msg["params"]["task_data"]["question"] == "What is 2+2?"
    
    def test_submission_from_jsonrpc(self):
        line = json.dumps({
            "jsonrpc": "2.0",
            "result": {"task_id": "test_1", "submission": "4"},
            "id": 1,
        })
        submission = Submission.from_jsonrpc(line)
        
        assert submission.task_id == "test_1"
        assert submission.answer == "4"
    
    def test_submission_error(self):
        line = json.dumps({
            "jsonrpc": "2.0",
            "error": {"code": -1, "message": "Something failed"},
            "id": 1,
        })
        
        with pytest.raises(RuntimeError, match="Agent error"):
            Submission.from_jsonrpc(line)


class TestGraders:
    """Test grading functions."""
    
    def test_exact_match(self):
        assert exact_match("hello", "hello")
        assert exact_match("  hello  ", "hello")
        assert not exact_match("Hello", "hello")
    
    def test_normalized_match(self):
        assert normalized_match("Hello World", "hello world")
        assert normalized_match("  hello   world  ", "hello world")
        assert not normalized_match("hello", "world")
    
    def test_numeric_match(self):
        assert numeric_match("42", "42")
        assert numeric_match("42.0", "42")
        assert numeric_match("The answer is 42", "42")
        assert numeric_match("2.15", "2.1498", rtol=0.01)
        # Very different numbers should not match
        assert not numeric_match("42", "100")
        assert not numeric_match("hello", "42")  # No number in first string
    
    def test_fuzzy_match(self):
        assert fuzzy_match("hello world", "hello world")
        assert fuzzy_match("hello world", "helo world", threshold=0.8)
        assert not fuzzy_match("hello", "goodbye")
    
    def test_grade_pipeline(self):
        # Exact match wins
        passed, method = grade_with_pipeline("4", "4")
        assert passed
        assert method == "exact"
        
        # Normalized match
        passed, method = grade_with_pipeline("Hello World", "hello world")
        assert passed
        assert method == "normalized"
        
        # Numeric match
        passed, method = grade_with_pipeline("The answer is 42", "42")
        assert passed
        assert method == "numeric"
        
        # No match
        passed, method = grade_with_pipeline("wrong", "correct")
        assert not passed
        assert method == "none"


class TestAgentRunner:
    """Test agent runner with echo agent."""
    
    def test_run_echo_agent(self, tmp_path):
        """Test running the echo agent."""
        # Get path to echo agent
        agent_path = Path(__file__).parent.parent / "agents/echo_agent.py"
        
        if not agent_path.exists():
            pytest.skip("Echo agent not found")
        
        runner = AgentRunner(agent_path)
        task = Task(id="test_1", data={"x": 42})
        
        result = runner.run(task, timeout=10)
        
        assert result.task_id == "test_1"
        # Echo agent returns the task data as JSON
        assert '"x": 42' in result.answer or '"x":42' in result.answer


class TestTraceLogger:
    """Test trace logging."""
    
    def test_log_to_file(self, tmp_path):
        trace_file = tmp_path / "trace.jsonl"
        
        with TraceLogger(trace_file, run_id="test_run") as logger:
            logger.log("task_start", task_id="t1")
            logger.log_raw({"type": "custom", "data": 123})
        
        lines = trace_file.read_text().strip().split("\n")
        assert len(lines) == 2
        
        first = json.loads(lines[0])
        assert first["type"] == "task_start"
        assert first["task_id"] == "t1"
        assert first["run_id"] == "test_run"
        
        second = json.loads(lines[1])
        assert second["type"] == "custom"
        assert second["data"] == 123


class TestErrorClassification:
    """Test error classification for retry logic."""

    def test_timeout_not_retryable(self):
        """Timeouts should NOT be retried (preserves pass@1 accuracy)."""
        from harness.parallel import classify_error, ErrorType

        # Various timeout error messages
        timeout_errors = [
            Exception("Agent timed out after 300s"),
            Exception("Request timed out"),
            Exception("timeout error"),
            Exception("Task TIMED OUT"),
        ]

        for err in timeout_errors:
            error_type, _ = classify_error(err)
            assert error_type == ErrorType.NON_RETRYABLE, f"Timeout should not retry: {err}"

    def test_rate_limit_retryable(self):
        """Rate limits (429) should be retried."""
        from harness.parallel import classify_error, ErrorType

        rate_limit_errors = [
            Exception("Error 429: Too many requests"),
            Exception("rate limit exceeded"),
            Exception("Rate Limit Error"),
        ]

        for err in rate_limit_errors:
            error_type, _ = classify_error(err)
            assert error_type == ErrorType.RETRYABLE, f"Rate limit should retry: {err}"

    def test_server_error_retryable(self):
        """Server errors (5xx) should be retried."""
        from harness.parallel import classify_error, ErrorType

        server_errors = [
            Exception("HTTP 500 Internal Server Error"),
            Exception("502 Bad Gateway"),
            Exception("503 Service Unavailable"),
            Exception("504 Gateway Timeout"),
        ]

        for err in server_errors:
            error_type, _ = classify_error(err)
            assert error_type == ErrorType.RETRYABLE, f"Server error should retry: {err}"

    def test_auth_error_not_retryable(self):
        """Auth errors should NOT be retried."""
        from harness.parallel import classify_error, ErrorType

        auth_errors = [
            Exception("401 Unauthorized"),
            Exception("403 Forbidden"),
            Exception("Invalid API key"),
        ]

        for err in auth_errors:
            error_type, _ = classify_error(err)
            assert error_type == ErrorType.NON_RETRYABLE, f"Auth error should not retry: {err}"

"""Tests for the 'harness continue' command."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from harness.cli import cli, find_run_dir, load_status_file, recover_run_state


# Path to the echo agent for integration tests
ECHO_AGENT = str(Path(__file__).parent.parent / "agents" / "echo_agent.py")


class TestFindRunDir:
    """Test run directory discovery."""

    def test_exact_match(self, tmp_path):
        """Find run dir by exact run_id match."""
        run_dir = tmp_path / "arithmetic" / "my-run-id"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text("{}")

        result = find_run_dir("my-run-id", tmp_path)
        assert result == run_dir

    def test_partial_match(self, tmp_path):
        """Find run dir by partial substring match."""
        run_dir = tmp_path / "arithmetic" / "arithmetic_agent_20260204_225130_5d8519"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text("{}")

        result = find_run_dir("5d8519", tmp_path)
        assert result == run_dir

    def test_direct_path(self, tmp_path):
        """Find run dir when given a direct path."""
        run_dir = tmp_path / "my-custom-results"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text("{}")

        result = find_run_dir(str(run_dir), tmp_path)
        assert result == run_dir

    def test_no_match(self, tmp_path):
        """Return None when no match is found."""
        result = find_run_dir("nonexistent", tmp_path)
        assert result is None

    def test_no_run_json(self, tmp_path):
        """Directories without run.json are not matched."""
        run_dir = tmp_path / "arithmetic" / "my-run-id"
        run_dir.mkdir(parents=True)
        # No run.json

        result = find_run_dir("my-run-id", tmp_path)
        assert result is None

    def test_multiple_partial_matches_picks_latest(self, tmp_path):
        """When multiple partial matches, pick most recent (last alphabetically)."""
        run1 = tmp_path / "arithmetic" / "arithmetic_agent_20260101_000000_aaa111"
        run2 = tmp_path / "arithmetic" / "arithmetic_agent_20260102_000000_aaa222"
        for d in [run1, run2]:
            d.mkdir(parents=True)
            (d / "run.json").write_text("{}")

        result = find_run_dir("arithmetic_agent", tmp_path)
        assert result == run2  # Most recent by timestamp in name


class TestContinueNoErrors:
    """Test continue when there are no errored tasks."""

    def test_no_errors_exits_cleanly(self, tmp_path):
        """When no errored tasks, print message and exit."""
        run_dir = tmp_path / "arithmetic" / "test-run"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": "test-run",
            "agent": ECHO_AGENT,
            "benchmark": "arithmetic",
            "model": None,
            "grader": "default",
            "parallel": 1,
            "max_retries": 1,
            "task_timeout": 30,
            "num_tasks_requested": 2,
            "error_task_ids": [],
            "completed_task_ids": ["arith_000", "arith_001"],
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["continue", "test-run", "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert "No errored or incomplete tasks" in result.output


class TestContinueValidation:
    """Test validation and error handling."""

    def test_missing_run_id(self, tmp_path):
        """Fail when run_id is not found."""
        runner = CliRunner()
        result = runner.invoke(cli, ["continue", "nonexistent", "-o", str(tmp_path)])
        assert result.exit_code != 0
        assert "Could not find run directory" in result.output

    def test_missing_agent(self, tmp_path):
        """Fail when agent path doesn't exist."""
        run_dir = tmp_path / "arithmetic" / "test-run"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": "test-run",
            "agent": "/nonexistent/agent.py",
            "benchmark": "arithmetic",
            "model": None,
            "grader": "default",
            "parallel": 1,
            "max_retries": 1,
            "task_timeout": 30,
            "error_task_ids": ["arith_000"],
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["continue", "test-run", "-o", str(tmp_path)])
        assert result.exit_code != 0
        assert "Agent not found" in result.output

    def test_no_benchmark(self, tmp_path):
        """Fail when original run had no benchmark."""
        run_dir = tmp_path / "custom" / "test-run"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": "test-run",
            "agent": ECHO_AGENT,
            "benchmark": None,
            "model": None,
            "grader": "default",
            "parallel": 1,
            "max_retries": 1,
            "task_timeout": 30,
            "error_task_ids": ["t1"],
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["continue", "test-run", "-o", str(tmp_path)])
        assert result.exit_code != 0
        assert "no benchmark" in result.output.lower()


class TestContinueIntegration:
    """Integration tests for the full continue flow."""

    def _create_failed_run(self, base_dir: Path, error_task_ids: list[str],
                            success_task_ids: list[str] | None = None) -> Path:
        """Create a fake failed run with the given error task IDs."""
        success_task_ids = success_task_ids or []

        run_dir = base_dir / "arithmetic" / "test-failed-run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build summary results
        results_data = []
        for tid in success_task_ids:
            results_data.append({
                "task_id": tid,
                "status": "completed",
                "submission": "42",
                "error": None,
                "attempts": 1,
                "duration_ms": 100,
            })
        for tid in error_task_ids:
            results_data.append({
                "task_id": tid,
                "status": "errored",
                "submission": None,
                "error": "Agent timed out",
                "attempts": 3,
                "duration_ms": 5000,
            })

        # Write run.json
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": "test-failed-run",
            "agent": ECHO_AGENT,
            "benchmark": "arithmetic",
            "model": None,
            "grader": "default",
            "parallel": 1,
            "max_retries": 1,
            "task_timeout": 30,
            "num_tasks_requested": len(success_task_ids) + len(error_task_ids),
            "error_task_ids": error_task_ids,
            "completed_task_ids": success_task_ids,
        }))

        # Write summary.json
        (run_dir / "summary.json").write_text(json.dumps({
            "benchmark": "arithmetic",
            "model": None,
            "total": len(results_data),
            "completed": len(success_task_ids),
            "errored": len(error_task_ids),
            "results": results_data,
        }))

        # Write trace files for successful tasks
        for tid in success_task_ids:
            trace_path = run_dir / f"trace_{tid}.jsonl"
            trace_path.write_text(json.dumps({
                "timestamp": 1234567890.0,
                "run_id": tid,
                "type": "task_complete",
                "task_id": tid,
                "submission": "42",
            }) + "\n")

        # Write (empty) trace files for errored tasks
        for tid in error_task_ids:
            trace_path = run_dir / f"trace_{tid}.jsonl"
            trace_path.write_text(json.dumps({
                "timestamp": 1234567890.0,
                "run_id": tid,
                "type": "task_error",
                "task_id": tid,
                "error": "timed out",
            }) + "\n")

        return run_dir

    def test_continue_reruns_errored_tasks(self, tmp_path):
        """Continue re-runs errored tasks and updates output files."""
        run_dir = self._create_failed_run(
            tmp_path,
            error_task_ids=["arith_000"],
            success_task_ids=["arith_001"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "continue", "test-failed-run",
            "-o", str(tmp_path),
        ])

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "Re-running 1 task" in result.output

        # Verify run.json was updated
        run_data = json.loads((run_dir / "run.json").read_text())
        assert run_data["run_id"] == "test-failed-run"
        # The echo agent should succeed, so error count should go down
        assert run_data["num_tasks_errored"] == 0
        assert "arith_000" in run_data["completed_task_ids"]
        assert "arith_001" in run_data["completed_task_ids"]

        # Verify summary.json was updated
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["completed"] == 2
        assert summary["errored"] == 0

        # Verify trace file was regenerated for errored task
        trace_path = run_dir / "trace_arith_000.jsonl"
        assert trace_path.exists()
        content = trace_path.read_text()
        # Old error trace should be replaced
        assert "task_error" not in content or "task_complete" in content

        # Verify grades.json was created (multi-grader format)
        grades_path = run_dir / "grades.json"
        assert grades_path.exists()
        grades = json.loads(grades_path.read_text())
        # New format: dict of grader_name -> list
        assert isinstance(grades, dict)
        # Should have at least one grader key
        first_key = next(iter(grades))
        assert len(grades[first_key]) == 2  # Both tasks graded

    def test_continue_preserves_successful_results(self, tmp_path):
        """Previously successful tasks are preserved in the merged output."""
        run_dir = self._create_failed_run(
            tmp_path,
            error_task_ids=["arith_002"],
            success_task_ids=["arith_000", "arith_001"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "continue", "test-failed-run",
            "-o", str(tmp_path),
        ])

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"

        # Check summary has all 3 tasks
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["total"] == 3
        task_ids = {r["task_id"] for r in summary["results"]}
        assert task_ids == {"arith_000", "arith_001", "arith_002"}

    def test_continue_old_format_compat(self, tmp_path):
        """Continue works with old run.json format (failed_task_ids instead of error_task_ids)."""
        run_dir = tmp_path / "arithmetic" / "old-format-run"
        run_dir.mkdir(parents=True)

        # Old format: uses failed_task_ids for execution errors, no error_task_ids field
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": "old-format-run",
            "agent": ECHO_AGENT,
            "benchmark": "arithmetic",
            "model": None,
            "grader": "default",
            "parallel": 1,
            "max_retries": 1,
            "task_timeout": 30,
            "num_tasks_requested": 2,
            "num_tasks_success": 1,
            "num_tasks_failed": 1,
            "successful_task_ids": ["arith_000"],
            "failed_task_ids": ["arith_001"],
        }))

        (run_dir / "summary.json").write_text(json.dumps({
            "benchmark": "arithmetic",
            "model": None,
            "total": 2,
            "completed": 1,
            "errored": 1,
            "results": [
                {"task_id": "arith_000", "status": "completed", "submission": "42",
                 "error": None, "attempts": 1, "duration_ms": 100},
                {"task_id": "arith_001", "status": "errored", "submission": None,
                 "error": "timed out", "attempts": 3, "duration_ms": 5000},
            ],
        }))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "continue", "old-format-run",
            "-o", str(tmp_path),
        ])

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "Re-running 1 task" in result.output

    def test_continue_with_overrides(self, tmp_path):
        """CLI overrides for parallel, max-retries, task-timeout are applied."""
        run_dir = self._create_failed_run(
            tmp_path,
            error_task_ids=["arith_000"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "continue", "test-failed-run",
            "-o", str(tmp_path),
            "-p", "5",
            "--max-retries", "1",
            "--task-timeout", "10",
        ])

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "Parallel: 5" in result.output
        assert "Retries: 1" in result.output
        assert "Timeout: 10s" in result.output

    def test_continue_deletes_old_traces(self, tmp_path):
        """Old trace files for errored tasks are deleted before re-running."""
        run_dir = self._create_failed_run(
            tmp_path,
            error_task_ids=["arith_000"],
        )

        old_trace = run_dir / "trace_arith_000.jsonl"
        assert old_trace.exists()
        old_content = old_trace.read_text()
        assert "task_error" in old_content

        runner = CliRunner()
        result = runner.invoke(cli, [
            "continue", "test-failed-run",
            "-o", str(tmp_path),
        ])

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"

        # Trace file should have new content (from the echo agent's successful run)
        new_content = old_trace.read_text()
        # The content should be different (new run data)
        assert new_content != old_content

    def test_continue_via_direct_path(self, tmp_path):
        """Continue works when given a direct path to the run directory."""
        run_dir = self._create_failed_run(
            tmp_path,
            error_task_ids=["arith_000"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "continue", str(run_dir),
            "-o", str(tmp_path),
        ])

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "Re-running 1 task" in result.output


# =========================================================================
# Tests for status.jsonl, run_config.json, and interrupted run recovery
# =========================================================================


class TestLoadStatusFile:
    """Test load_status_file helper."""

    def test_empty_file(self, tmp_path):
        """Returns empty dict for empty file."""
        status_path = tmp_path / "status.jsonl"
        status_path.write_text("")
        assert load_status_file(status_path) == {}

    def test_missing_file(self, tmp_path):
        """Returns empty dict for missing file."""
        assert load_status_file(tmp_path / "status.jsonl") == {}

    def test_parses_entries(self, tmp_path):
        """Correctly parses JSONL entries."""
        status_path = tmp_path / "status.jsonl"
        status_path.write_text(
            json.dumps({"task_id": "t1", "status": "completed", "submission": "42"}) + "\n"
            + json.dumps({"task_id": "t2", "status": "errored", "error": "timeout"}) + "\n"
        )
        result = load_status_file(status_path)
        assert len(result) == 2
        assert result["t1"]["status"] == "completed"
        assert result["t2"]["status"] == "errored"

    def test_last_entry_wins(self, tmp_path):
        """When a task appears multiple times, last entry wins."""
        status_path = tmp_path / "status.jsonl"
        status_path.write_text(
            json.dumps({"task_id": "t1", "status": "errored", "error": "timeout"}) + "\n"
            + json.dumps({"task_id": "t1", "status": "completed", "submission": "42"}) + "\n"
        )
        result = load_status_file(status_path)
        assert result["t1"]["status"] == "completed"

    def test_skips_bad_lines(self, tmp_path):
        """Malformed lines are silently skipped."""
        status_path = tmp_path / "status.jsonl"
        status_path.write_text(
            "not json\n"
            + json.dumps({"task_id": "t1", "status": "completed"}) + "\n"
            + "{bad json\n"
        )
        result = load_status_file(status_path)
        assert len(result) == 1
        assert "t1" in result


class TestRecoverRunState:
    """Test recover_run_state helper."""

    def test_from_run_json(self, tmp_path):
        """Recovers state from run.json (completed run)."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": "run1",
            "agent": "my-agent",
            "benchmark": "arithmetic",
            "error_task_ids": ["t2"],
            "completed_task_ids": ["t1"],
        }))
        state = recover_run_state(run_dir)
        assert state["run_id"] == "run1"
        assert state["error_task_ids"] == ["t2"]
        assert state["completed_task_ids"] == ["t1"]
        assert state["incomplete_task_ids"] == []

    def test_from_run_json_old_format(self, tmp_path):
        """Recovers state from old run.json with failed_task_ids."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": "run1",
            "agent": "my-agent",
            "benchmark": "arithmetic",
            "failed_task_ids": ["t2"],
            "completed_task_ids": ["t1"],
        }))
        state = recover_run_state(run_dir)
        assert state["error_task_ids"] == ["t2"]

    def test_from_config_and_status(self, tmp_path):
        """Recovers state from run_config.json + status.jsonl."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        (run_dir / "run_config.json").write_text(json.dumps({
            "run_id": "run1",
            "agent": "my-agent",
            "benchmark": "gaia",
            "model": "deepseek",
            "task_ids": ["t1", "t2", "t3"],
        }))
        (run_dir / "status.jsonl").write_text(
            json.dumps({"task_id": "t1", "status": "completed", "submission": "42"}) + "\n"
            + json.dumps({"task_id": "t2", "status": "errored", "error": "crash"}) + "\n"
        )
        state = recover_run_state(run_dir)
        assert state["run_id"] == "run1"
        assert state["benchmark"] == "gaia"
        assert state["completed_task_ids"] == ["t1"]
        assert state["error_task_ids"] == ["t2"]
        assert state["incomplete_task_ids"] == ["t3"]

    def test_from_status_only(self, tmp_path):
        """Recovers from status.jsonl even without run_config.json."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        (run_dir / "status.jsonl").write_text(
            json.dumps({"task_id": "t1", "status": "completed", "submission": "42"}) + "\n"
        )
        state = recover_run_state(run_dir)
        assert state["completed_task_ids"] == ["t1"]
        assert state["incomplete_task_ids"] == []

    def test_discovers_tasks_from_traces(self, tmp_path):
        """Finds tasks that have trace files but no status entry.

        t2 has task_start but no task_complete → classified as errored (interrupted).
        t3 has an empty trace → classified as incomplete (never started).
        """
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        (run_dir / "run_config.json").write_text(json.dumps({
            "run_id": "run1",
            "agent": "my-agent",
            "benchmark": "gaia",
            "task_ids": ["t1", "t2", "t3"],
        }))
        # Only t1 succeeded
        (run_dir / "status.jsonl").write_text(
            json.dumps({"task_id": "t1", "status": "completed", "submission": "42"}) + "\n"
        )
        # t2 has a trace file (was in-progress) but no status entry
        (run_dir / "trace_t2.jsonl").write_text(
            json.dumps({"type": "task_start", "task_id": "t2"}) + "\n"
        )
        # t3 has an empty trace file
        (run_dir / "trace_t3.jsonl").write_text("")

        state = recover_run_state(run_dir)
        assert state["completed_task_ids"] == ["t1"]
        # t2 was interrupted (has start, no completion) → errored
        assert "t2" in state["error_task_ids"]
        # t3 has empty trace → incomplete
        assert "t3" in state["incomplete_task_ids"]

    def test_trace_scan_task_complete(self, tmp_path):
        """Trace scanning picks up task_complete events as success."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        lines = [
            json.dumps({"type": "task_start", "task_id": "t1"}),
            json.dumps({"type": "agent_stdout", "content": "some output"}),
            json.dumps({"type": "task_complete", "task_id": "t1", "submission": "42", "duration_ms": 5000}),
        ]
        (run_dir / "trace_t1.jsonl").write_text("\n".join(lines) + "\n")

        state = recover_run_state(run_dir)
        assert state["completed_task_ids"] == ["t1"]
        assert state["error_task_ids"] == []
        assert state["incomplete_task_ids"] == []
        # Verify submission is captured
        result = [r for r in state["results"] if r["task_id"] == "t1"][0]
        assert result["submission"] == "42"
        assert result["duration_ms"] == 5000

    def test_trace_scan_task_error(self, tmp_path):
        """Trace scanning picks up task_error events as failed."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        lines = [
            json.dumps({"type": "task_start", "task_id": "t1"}),
            json.dumps({"type": "task_error", "task_id": "t1", "error": "boom"}),
        ]
        (run_dir / "trace_t1.jsonl").write_text("\n".join(lines) + "\n")

        state = recover_run_state(run_dir)
        assert state["completed_task_ids"] == []
        assert state["error_task_ids"] == ["t1"]
        result = [r for r in state["results"] if r["task_id"] == "t1"][0]
        assert result["error"] == "boom"
        assert result["status"] == "errored"

    def test_trace_scan_interrupted(self, tmp_path):
        """Trace with task_start but no completion is 'interrupted'."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        lines = [
            json.dumps({"type": "task_start", "task_id": "t1"}),
            json.dumps({"type": "agent_stdout", "content": "partial work"}),
        ]
        (run_dir / "trace_t1.jsonl").write_text("\n".join(lines) + "\n")

        state = recover_run_state(run_dir)
        assert state["completed_task_ids"] == []
        assert state["error_task_ids"] == ["t1"]
        result = [r for r in state["results"] if r["task_id"] == "t1"][0]
        assert result["status"] == "interrupted"

    def test_status_takes_priority_over_trace_scan(self, tmp_path):
        """status.jsonl entries override trace scanning results."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        # status.jsonl says completed
        (run_dir / "status.jsonl").write_text(
            json.dumps({"task_id": "t1", "status": "completed", "submission": "correct"}) + "\n"
        )
        # trace file still has only task_start (would be interrupted)
        lines = [
            json.dumps({"type": "task_start", "task_id": "t1"}),
        ]
        (run_dir / "trace_t1.jsonl").write_text("\n".join(lines) + "\n")

        state = recover_run_state(run_dir)
        assert state["completed_task_ids"] == ["t1"]
        assert state["error_task_ids"] == []

    def test_mixed_trace_statuses(self, tmp_path):
        """Mix of completed, errored, interrupted, and empty traces."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        # t1: completed
        (run_dir / "trace_t1.jsonl").write_text(
            json.dumps({"type": "task_start", "task_id": "t1"}) + "\n" +
            json.dumps({"type": "task_complete", "task_id": "t1", "submission": "a"}) + "\n"
        )
        # t2: errored
        (run_dir / "trace_t2.jsonl").write_text(
            json.dumps({"type": "task_start", "task_id": "t2"}) + "\n" +
            json.dumps({"type": "task_error", "task_id": "t2", "error": "fail"}) + "\n"
        )
        # t3: interrupted
        (run_dir / "trace_t3.jsonl").write_text(
            json.dumps({"type": "task_start", "task_id": "t3"}) + "\n"
        )
        # t4: empty (never started)
        (run_dir / "trace_t4.jsonl").write_text("")

        state = recover_run_state(run_dir)
        assert state["completed_task_ids"] == ["t1"]
        assert sorted(state["error_task_ids"]) == ["t2", "t3"]
        assert state["incomplete_task_ids"] == ["t4"]

    def test_empty_dir(self, tmp_path):
        """Handles a directory with nothing useful."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        state = recover_run_state(run_dir)
        assert state["completed_task_ids"] == []
        assert state["error_task_ids"] == []
        assert state["incomplete_task_ids"] == []


class TestFindRunDirEnhanced:
    """Test find_run_dir with status.jsonl and run_config.json."""

    def test_finds_dir_with_status_jsonl(self, tmp_path):
        """Find run dir that only has status.jsonl (no run.json)."""
        run_dir = tmp_path / "gaia" / "gaia-run-123"
        run_dir.mkdir(parents=True)
        (run_dir / "status.jsonl").write_text(
            json.dumps({"task_id": "t1", "status": "completed"}) + "\n"
        )
        result = find_run_dir("gaia-run-123", tmp_path)
        assert result == run_dir

    def test_finds_dir_with_run_config(self, tmp_path):
        """Find run dir that only has run_config.json."""
        run_dir = tmp_path / "gaia" / "gaia-run-456"
        run_dir.mkdir(parents=True)
        (run_dir / "run_config.json").write_text("{}")
        result = find_run_dir("gaia-run-456", tmp_path)
        assert result == run_dir

    def test_partial_match_status_only(self, tmp_path):
        """Partial match works with status.jsonl only."""
        run_dir = tmp_path / "gaia" / "gaia_agent_20260206_abc123"
        run_dir.mkdir(parents=True)
        (run_dir / "status.jsonl").write_text("")
        result = find_run_dir("abc123", tmp_path)
        assert result == run_dir

    def test_direct_path_status_only(self, tmp_path):
        """Direct path works with status.jsonl only."""
        run_dir = tmp_path / "my-run"
        run_dir.mkdir(parents=True)
        (run_dir / "status.jsonl").write_text("")
        result = find_run_dir(str(run_dir), tmp_path)
        assert result == run_dir

    def test_dir_without_any_marker_not_found(self, tmp_path):
        """Dir without run.json, run_config.json, status.jsonl, or traces is not found."""
        run_dir = tmp_path / "gaia" / "gaia-run-789"
        run_dir.mkdir(parents=True)
        (run_dir / "readme.txt").write_text("not a run")
        result = find_run_dir("gaia-run-789", tmp_path)
        assert result is None

    def test_finds_dir_with_traces_only(self, tmp_path):
        """Find run dir that only has trace files (old interrupted run)."""
        run_dir = tmp_path / "gaia" / "gaia-run-old"
        run_dir.mkdir(parents=True)
        (run_dir / "trace_t1.jsonl").write_text("")
        result = find_run_dir("gaia-run-old", tmp_path)
        assert result == run_dir


class TestStatusFileWriting:
    """Test that ParallelRunner writes status.jsonl."""

    def test_status_written_on_success(self, tmp_path):
        """status.jsonl entry is created when a task succeeds."""
        from harness.parallel import ParallelRunner, RetryConfig

        runner = ParallelRunner(
            agent_path=Path(ECHO_AGENT),
            output_dir=tmp_path,
            max_parallel=1,
            retry_config=RetryConfig(max_retries=1),
            task_timeout=30,
        )

        from harness.protocol import Task
        import asyncio

        task = Task(id="test_task", data={"question": "2+2"})
        results = asyncio.run(runner.run_all([task]))

        assert len(results) == 1
        assert results[0].status == "completed"

        # Check status.jsonl was written
        status_path = tmp_path / "status.jsonl"
        assert status_path.exists()
        status_map = load_status_file(status_path)
        assert "test_task" in status_map
        assert status_map["test_task"]["status"] == "completed"
        assert "timestamp" in status_map["test_task"]

    def test_status_written_on_failure(self, tmp_path):
        """status.jsonl entry is created when a task fails."""
        from harness.parallel import ParallelRunner, RetryConfig, TaskResult

        runner = ParallelRunner(
            agent_path=Path("/nonexistent/agent.py"),
            output_dir=tmp_path,
            max_parallel=1,
            retry_config=RetryConfig(max_retries=1),
            task_timeout=5,
        )

        from harness.protocol import Task
        import asyncio

        task = Task(id="fail_task", data={"question": "test"})
        results = asyncio.run(runner.run_all([task]))

        assert len(results) == 1
        assert results[0].status == "errored"

        status_path = tmp_path / "status.jsonl"
        assert status_path.exists()
        status_map = load_status_file(status_path)
        assert "fail_task" in status_map
        assert status_map["fail_task"]["status"] == "errored"


class TestContinueInterruptedRun:
    """Integration tests for continuing interrupted runs (no run.json)."""

    def _create_interrupted_run(self, base_dir: Path) -> Path:
        """Create a fake interrupted run with run_config.json and partial status.jsonl."""
        run_dir = base_dir / "arithmetic" / "test-interrupted-run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write run_config (saved at start of run)
        (run_dir / "run_config.json").write_text(json.dumps({
            "run_id": "test-interrupted-run",
            "agent": ECHO_AGENT,
            "benchmark": "arithmetic",
            "model": None,
            "parallel": 2,
            "max_retries": 1,
            "task_timeout": 30,
            "total_tasks": 5,
            "task_ids": ["arith_000", "arith_001", "arith_002", "arith_003", "arith_004"],
        }))

        # Only 2 out of 5 tasks completed before kill
        (run_dir / "status.jsonl").write_text(
            json.dumps({"task_id": "arith_000", "status": "completed", "submission": "42", "attempts": 1, "duration_ms": 100}) + "\n"
            + json.dumps({"task_id": "arith_001", "status": "completed", "submission": "42", "attempts": 1, "duration_ms": 150}) + "\n"
        )

        # Trace files for completed tasks
        for tid in ["arith_000", "arith_001"]:
            (run_dir / f"trace_{tid}.jsonl").write_text(
                json.dumps({"type": "task_complete", "task_id": tid}) + "\n"
            )
        # In-progress trace (started but not finished)
        (run_dir / "trace_arith_002.jsonl").write_text(
            json.dumps({"type": "task_start", "task_id": "arith_002"}) + "\n"
        )
        # Empty trace (created but process killed immediately)
        (run_dir / "trace_arith_003.jsonl").write_text("")

        return run_dir

    def test_continue_interrupted_run(self, tmp_path):
        """Continue picks up incomplete tasks from an interrupted run.

        arith_002 has task_start but no completion → errored (interrupted)
        arith_003 has empty trace → incomplete
        arith_004 has no trace at all → incomplete
        """
        run_dir = self._create_interrupted_run(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "continue", "test-interrupted-run",
            "-o", str(tmp_path),
        ])

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "Completed: 2" in result.output
        assert "Errored: 1" in result.output
        assert "Incomplete: 2" in result.output
        assert "Re-running 3 task" in result.output

        # run.json should now exist
        assert (run_dir / "run.json").exists()

        # summary.json should have all 5 tasks
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["total"] == 5
        assert summary["completed"] == 5  # echo agent completes everything

    def test_continue_interrupted_preserves_completed(self, tmp_path):
        """Previously completed tasks are not re-run."""
        run_dir = self._create_interrupted_run(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "continue", "test-interrupted-run",
            "-o", str(tmp_path),
        ])

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"

        # The original completed tasks should still have their original traces
        # (not overwritten by re-running)
        summary = json.loads((run_dir / "summary.json").read_text())
        completed_ids = {r["task_id"] for r in summary["results"] if r["status"] == "completed"}
        assert "arith_000" in completed_ids
        assert "arith_001" in completed_ids

    def test_continue_interrupted_run_overrides(self, tmp_path):
        """CLI overrides work for interrupted runs."""
        run_dir = self._create_interrupted_run(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "continue", "test-interrupted-run",
            "-o", str(tmp_path),
            "-p", "10",
            "--task-timeout", "60",
        ])

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "Parallel: 10" in result.output
        assert "Timeout: 60s" in result.output

    def test_continue_trace_only_with_cli_overrides(self, tmp_path):
        """Continue an old interrupted run that only has trace files.

        Requires --agent and --benchmark supplied via CLI since there's no
        run_config.json or run.json.
        """
        run_dir = tmp_path / "arithmetic" / "trace-only-run"
        run_dir.mkdir(parents=True)

        # Create trace files mimicking an old interrupted run:
        # arith_000: completed
        (run_dir / "trace_arith_000.jsonl").write_text(
            json.dumps({"type": "task_start", "task_id": "arith_000"}) + "\n" +
            json.dumps({"type": "task_complete", "task_id": "arith_000", "submission": "42", "duration_ms": 1000}) + "\n"
        )
        # arith_001: interrupted (started but no completion)
        (run_dir / "trace_arith_001.jsonl").write_text(
            json.dumps({"type": "task_start", "task_id": "arith_001"}) + "\n"
        )
        # arith_002: empty trace (never started)
        (run_dir / "trace_arith_002.jsonl").write_text("")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "continue", str(run_dir),
            "--agent", ECHO_AGENT,
            "--benchmark", "arithmetic",
        ])

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "Completed: 1" in result.output
        # arith_001 errored (interrupted) + arith_002 incomplete + discovered from benchmark
        assert "Discovered" in result.output
        assert (run_dir / "summary.json").exists()

    def test_continue_no_agent_errors(self, tmp_path):
        """Continue fails if no agent can be determined."""
        run_dir = tmp_path / "gaia" / "mystery-run"
        run_dir.mkdir(parents=True)
        (run_dir / "trace_t1.jsonl").write_text("")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "continue", str(run_dir),
            "--benchmark", "arithmetic",
        ])

        assert result.exit_code != 0
        assert "no agent specified" in result.output.lower() or "no agent specified" in (result.output + (result.exception and str(result.exception) or "")).lower()

    def test_continue_no_benchmark_errors(self, tmp_path):
        """Continue fails if no benchmark can be determined."""
        run_dir = tmp_path / "gaia" / "mystery-run2"
        run_dir.mkdir(parents=True)
        (run_dir / "trace_t1.jsonl").write_text("")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "continue", str(run_dir),
            "--agent", ECHO_AGENT,
        ])

        assert result.exit_code != 0
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
                "status": "success",
                "submission": "42",
                "error": None,
                "attempts": 1,
                "duration_ms": 100,
            })
        for tid in error_task_ids:
            results_data.append({
                "task_id": tid,
                "status": "failed",
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
            "success": len(success_task_ids),
            "failed": len(error_task_ids),
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
        assert summary["success"] == 2
        assert summary["failed"] == 0

        # Verify trace file was regenerated for errored task
        trace_path = run_dir / "trace_arith_000.jsonl"
        assert trace_path.exists()
        content = trace_path.read_text()
        # Old error trace should be replaced
        assert "task_error" not in content or "task_complete" in content

        # Verify grades.json was created
        grades_path = run_dir / "grades.json"
        assert grades_path.exists()
        grades = json.loads(grades_path.read_text())
        assert len(grades) == 2  # Both tasks graded

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
            "success": 1,
            "failed": 1,
            "results": [
                {"task_id": "arith_000", "status": "success", "submission": "42",
                 "error": None, "attempts": 1, "duration_ms": 100},
                {"task_id": "arith_001", "status": "failed", "submission": None,
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
            json.dumps({"task_id": "t1", "status": "success", "submission": "42"}) + "\n"
            + json.dumps({"task_id": "t2", "status": "failed", "error": "timeout"}) + "\n"
        )
        result = load_status_file(status_path)
        assert len(result) == 2
        assert result["t1"]["status"] == "success"
        assert result["t2"]["status"] == "failed"

    def test_last_entry_wins(self, tmp_path):
        """When a task appears multiple times, last entry wins."""
        status_path = tmp_path / "status.jsonl"
        status_path.write_text(
            json.dumps({"task_id": "t1", "status": "failed", "error": "timeout"}) + "\n"
            + json.dumps({"task_id": "t1", "status": "success", "submission": "42"}) + "\n"
        )
        result = load_status_file(status_path)
        assert result["t1"]["status"] == "success"

    def test_skips_bad_lines(self, tmp_path):
        """Malformed lines are silently skipped."""
        status_path = tmp_path / "status.jsonl"
        status_path.write_text(
            "not json\n"
            + json.dumps({"task_id": "t1", "status": "success"}) + "\n"
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
            json.dumps({"task_id": "t1", "status": "success", "submission": "42"}) + "\n"
            + json.dumps({"task_id": "t2", "status": "failed", "error": "crash"}) + "\n"
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
            json.dumps({"task_id": "t1", "status": "success", "submission": "42"}) + "\n"
        )
        state = recover_run_state(run_dir)
        assert state["completed_task_ids"] == ["t1"]
        assert state["incomplete_task_ids"] == []

    def test_discovers_tasks_from_traces(self, tmp_path):
        """Finds tasks that have trace files but no status entry."""
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
            json.dumps({"task_id": "t1", "status": "success", "submission": "42"}) + "\n"
        )
        # t2 has a trace file (was in-progress) but no status entry
        (run_dir / "trace_t2.jsonl").write_text(
            json.dumps({"type": "task_start", "task_id": "t2"}) + "\n"
        )
        # t3 has an empty trace file
        (run_dir / "trace_t3.jsonl").write_text("")

        state = recover_run_state(run_dir)
        assert state["completed_task_ids"] == ["t1"]
        assert sorted(state["incomplete_task_ids"]) == ["t2", "t3"]

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
            json.dumps({"task_id": "t1", "status": "success"}) + "\n"
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
        """Dir without run.json, run_config.json, or status.jsonl is not found."""
        run_dir = tmp_path / "gaia" / "gaia-run-789"
        run_dir.mkdir(parents=True)
        (run_dir / "trace_t1.jsonl").write_text("")  # Only trace files
        result = find_run_dir("gaia-run-789", tmp_path)
        assert result is None


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
        assert results[0].status == "success"

        # Check status.jsonl was written
        status_path = tmp_path / "status.jsonl"
        assert status_path.exists()
        status_map = load_status_file(status_path)
        assert "test_task" in status_map
        assert status_map["test_task"]["status"] == "success"
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
        assert results[0].status == "failed"

        status_path = tmp_path / "status.jsonl"
        assert status_path.exists()
        status_map = load_status_file(status_path)
        assert "fail_task" in status_map
        assert status_map["fail_task"]["status"] == "failed"


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
            json.dumps({"task_id": "arith_000", "status": "success", "submission": "42", "attempts": 1, "duration_ms": 100}) + "\n"
            + json.dumps({"task_id": "arith_001", "status": "success", "submission": "42", "attempts": 1, "duration_ms": 150}) + "\n"
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
        """Continue picks up 3 incomplete tasks from an interrupted run."""
        run_dir = self._create_interrupted_run(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "continue", "test-interrupted-run",
            "-o", str(tmp_path),
        ])

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "Completed: 2" in result.output
        assert "Incomplete: 3" in result.output
        assert "Re-running 3 task" in result.output

        # run.json should now exist
        assert (run_dir / "run.json").exists()

        # summary.json should have all 5 tasks
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["total"] == 5
        assert summary["success"] == 5  # echo agent succeeds on everything

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
        completed_ids = {r["task_id"] for r in summary["results"] if r["status"] == "success"}
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
"""Tests for HuggingFace integration."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from harness.hf_integration import (
    _get_token,
    push_run,
    list_runs,
    pull_run,
    ensure_dataset_repo,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_run_data():
    """Sample run.json data."""
    return {
        "run_id": "test-run-001",
        "timestamp": "2026-02-05T10:00:00Z",
        "agent": "src/harness/examples/echo_agent.py",
        "benchmark": "arithmetic",
        "model": "deepseek/deepseek-chat-v3",
        "grader": "default",
        "parallel": 10,
        "max_retries": 3,
        "task_timeout": 300,
        "num_tasks_requested": 5,
        "num_tasks_run": 5,
        "num_tasks_completed": 4,
        "num_tasks_errored": 1,
        "completed_task_ids": ["t1", "t2", "t3", "t4"],
        "error_task_ids": ["t5"],
        "score": 75.0,
        "passed": 3,
        "total_graded": 4,
        "total_cost_usd": 0.0012,
        "total_latency_ms": 5000.0,
        "total_usage": {
            "prompt_tokens": 500,
            "completion_tokens": 100,
            "total_tokens": 600,
        },
        "duration_seconds": 5.0,
        "git_commit": "abc123",
        "git_branch": "main",
        "model_stats": {},
        "task_stats": [],
        "agent_metrics": {},
    }


@pytest.fixture
def run_dir(tmp_path, sample_run_data):
    """Create a temporary run directory with run.json."""
    run_path = tmp_path / "results" / "arithmetic" / "test-run-001"
    run_path.mkdir(parents=True)
    (run_path / "run.json").write_text(json.dumps(sample_run_data, indent=2))
    return run_path


# ── Token tests ───────────────────────────────────────────────────────

class TestGetToken:
    def test_reads_hf_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token-123")
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        assert _get_token() == "test-token-123"

    def test_reads_huggingface_hub_token(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "alt-token-456")
        assert _get_token() == "alt-token-456"

    def test_hf_token_takes_priority(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "primary-token")
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "secondary-token")
        assert _get_token() == "primary-token"

    def test_raises_without_token(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        with pytest.raises(ValueError, match="HuggingFace token not found"):
            _get_token()


# ── Push run tests (mocked HF API) ───────────────────────────────────

class TestPushRun:
    @patch("harness.hf_integration._get_api")
    def test_push_uploads_full_run_json(self, mock_get_api, run_dir, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")

        mock_api = MagicMock()
        mock_get_api.return_value = mock_api
        mock_api.create_repo.return_value = "https://huggingface.co/datasets/user/runs"
        mock_api.file_exists.return_value = True  # README exists

        result = push_run(
            run_dir=run_dir,
            repo_id="user/runs",
            token="test-token",
        )

        assert result["run_id"] == "test-run-001"
        assert "user/runs" in result["repo_url"]
        assert "test-run-001.json" in result["file_url"]

        # Verify upload_file was called with {run_id}.json (not in data/ subfolder)
        upload_calls = mock_api.upload_file.call_args_list
        run_upload = [c for c in upload_calls if "test-run-001.json" in str(c)]
        assert len(run_upload) >= 1
        # Verify it's at repo root, not data/
        call_kwargs = run_upload[0].kwargs if run_upload[0].kwargs else {}
        if "path_in_repo" in call_kwargs:
            assert call_kwargs["path_in_repo"] == "test-run-001.json"

    @patch("harness.hf_integration._get_api")
    def test_push_creates_repo_if_needed(self, mock_get_api, run_dir, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")

        mock_api = MagicMock()
        mock_get_api.return_value = mock_api
        mock_api.create_repo.return_value = "https://huggingface.co/datasets/user/new-runs"
        mock_api.file_exists.return_value = True

        push_run(run_dir=run_dir, repo_id="user/new-runs", token="test-token", private=True)

        mock_api.create_repo.assert_called_once_with(
            repo_id="user/new-runs",
            repo_type="dataset",
            exist_ok=True,
            private=True,
        )

    def test_push_missing_run_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No run.json"):
            push_run(run_dir=empty_dir, repo_id="user/runs")

    @patch("harness.hf_integration._get_api")
    def test_push_no_index_file(self, mock_get_api, run_dir, monkeypatch):
        """Push should NOT create any runs.jsonl index."""
        monkeypatch.setenv("HF_TOKEN", "test-token")

        mock_api = MagicMock()
        mock_get_api.return_value = mock_api
        mock_api.create_repo.return_value = "https://huggingface.co/datasets/user/runs"
        mock_api.file_exists.return_value = True

        push_run(run_dir=run_dir, repo_id="user/runs", token="test-token")

        # No upload should reference runs.jsonl
        for c in mock_api.upload_file.call_args_list:
            assert "runs.jsonl" not in str(c)


# ── List runs tests (mocked) ─────────────────────────────────────────

class TestListRuns:
    @patch("harness.hf_integration.hf_hub_download")
    @patch("harness.hf_integration._get_api")
    def test_list_runs_downloads_all_json(self, mock_get_api, mock_download, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")

        mock_api = MagicMock()
        mock_get_api.return_value = mock_api
        mock_api.list_repo_files.return_value = ["README.md", "run1.json", "run2.json"]

        # Create mock downloaded files
        f1 = tmp_path / "run1.json"
        f1.write_text(json.dumps({"run_id": "run1", "score": 80.0}))
        f2 = tmp_path / "run2.json"
        f2.write_text(json.dumps({"run_id": "run2", "score": 95.0}))

        mock_download.side_effect = [str(f1), str(f2)]

        runs = list_runs(repo_id="user/runs", token="test-token")

        assert len(runs) == 2
        assert runs[0]["run_id"] == "run1"
        assert runs[1]["score"] == 95.0
        # Should only download .json files (not README.md)
        assert mock_download.call_count == 2

    @patch("harness.hf_integration._get_api")
    def test_list_runs_empty_repo(self, mock_get_api, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")

        mock_api = MagicMock()
        mock_get_api.return_value = mock_api
        mock_api.list_repo_files.return_value = ["README.md"]

        runs = list_runs(repo_id="user/runs", token="test-token")
        assert runs == []

    @patch("harness.hf_integration._get_api")
    def test_list_runs_repo_not_found(self, mock_get_api, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")

        mock_api = MagicMock()
        mock_get_api.return_value = mock_api
        mock_api.list_repo_files.side_effect = Exception("not found")

        runs = list_runs(repo_id="user/runs", token="test-token")
        assert runs == []


# ── Pull run tests (mocked) ──────────────────────────────────────────

class TestPullRun:
    @patch("harness.hf_integration.hf_hub_download")
    def test_pull_downloads_to_output(self, mock_download, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")

        src = tmp_path / "source.json"
        src.write_text(json.dumps({"run_id": "run1"}))
        mock_download.return_value = str(src)

        output = tmp_path / "downloads"
        path = pull_run(run_id="run1", repo_id="user/runs", output_dir=output, token="test-token")

        assert path.exists()
        assert path.name == "run1.json"
        data = json.loads(path.read_text())
        assert data["run_id"] == "run1"

        # Verify filename is {run_id}.json (not data/{run_id}.json)
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["filename"] == "run1.json"

    @patch("harness.hf_integration.hf_hub_download")
    def test_pull_returns_cache_path_without_output(self, mock_download, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")

        cached = tmp_path / "cached.json"
        cached.write_text("{}")
        mock_download.return_value = str(cached)

        path = pull_run(run_id="run1", repo_id="user/runs", token="test-token")
        assert path == cached


# ── Ensure dataset repo tests ────────────────────────────────────────

class TestEnsureDatasetRepo:
    @patch("harness.hf_integration._get_api")
    def test_creates_repo(self, mock_get_api, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")
        mock_api = MagicMock()
        mock_get_api.return_value = mock_api
        mock_api.create_repo.return_value = "https://huggingface.co/datasets/user/runs"
        mock_api.file_exists.return_value = True

        url = ensure_dataset_repo("user/runs", token="test-token", private=True)

        assert "user/runs" in url
        mock_api.create_repo.assert_called_once_with(
            repo_id="user/runs",
            repo_type="dataset",
            exist_ok=True,
            private=True,
        )


# ── CLI integration tests ────────────────────────────────────────────

class TestCLI:
    def test_push_help(self):
        from click.testing import CliRunner
        from harness.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["push", "--help"])
        assert result.exit_code == 0
        assert "HuggingFace" in result.output

    def test_push_missing_repo(self):
        from click.testing import CliRunner
        from harness.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["push", "some-run"])
        assert result.exit_code != 0

    def test_pull_help(self):
        from click.testing import CliRunner
        from harness.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "--help"])
        assert result.exit_code == 0

    def test_runs_help(self):
        from click.testing import CliRunner
        from harness.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["runs", "--help"])
        assert result.exit_code == 0

    @patch("harness.hf_integration.push_run")
    def test_push_with_valid_run(self, mock_push, tmp_path):
        from click.testing import CliRunner
        from harness.cli import cli

        run_dir = tmp_path / "results" / "arithmetic" / "my-run"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({"run_id": "my-run"}))

        mock_push.return_value = {
            "run_id": "my-run",
            "repo_url": "https://huggingface.co/datasets/user/runs",
            "file_url": "https://huggingface.co/datasets/user/runs/blob/main/my-run.json",
        }

        runner = CliRunner()
        result = runner.invoke(cli, [
            "push", str(run_dir),
            "--repo", "user/runs",
            "--token", "test-token",
        ])

        assert result.exit_code == 0
        assert "my-run" in result.output
        mock_push.assert_called_once()

"""Tests for agent resolution and HAL generalist agent structure."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from harness.cli import resolve_agent_path


# ── Agent resolution tests ────────────────────────────────────────────

class TestResolveAgentPath:
    """Test resolve_agent_path resolves agents from various locations."""

    def test_exact_file_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agent = tmp_path / "my_agent.py"
        agent.write_text("# agent")
        result = resolve_agent_path(str(agent))
        assert result == agent

    def test_exact_dir_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agent_dir = tmp_path / "my_agent"
        agent_dir.mkdir()
        (agent_dir / "agent.py").write_text("# agent")
        result = resolve_agent_path(str(agent_dir))
        assert result == agent_dir

    def test_name_resolves_to_agents_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "echo_agent.py").write_text("# agent")
        result = resolve_agent_path("echo_agent.py")
        assert result.resolve() == (agents_dir / "echo_agent.py").resolve()

    def test_name_resolves_to_agents_subdir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agents_dir = tmp_path / "agents" / "hal_generalist"
        agents_dir.mkdir(parents=True)
        (agents_dir / "agent.py").write_text("# agent")
        result = resolve_agent_path("hal_generalist")
        # Should resolve to the directory (not agent.py inside)
        assert result.resolve() == agents_dir.resolve()

    def test_name_with_py_extension_resolves(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "my_agent.py").write_text("# agent")
        # Search without extension
        result = resolve_agent_path("my_agent")
        assert result.resolve() == (agents_dir / "my_agent.py").resolve()

    def test_not_found_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(Exception, match="Agent not found"):
            resolve_agent_path("nonexistent_agent")


# ── HAL generalist agent structure tests ──────────────────────────────

class TestHALGeneralistStructure:
    """Verify the HAL generalist agent has all required files."""

    @pytest.fixture
    def hal_dir(self):
        return Path(__file__).parent.parent / "agents" / "hal_generalist"

    def test_directory_exists(self, hal_dir):
        assert hal_dir.is_dir()

    def test_has_init(self, hal_dir):
        assert (hal_dir / "__init__.py").exists()

    def test_has_agent_py(self, hal_dir):
        assert (hal_dir / "agent.py").exists()

    def test_has_smolagents_runner(self, hal_dir):
        assert (hal_dir / "smolagents_runner.py").exists()

    def test_has_tools(self, hal_dir):
        assert (hal_dir / "tools.py").exists()

    def test_has_prompts(self, hal_dir):
        assert (hal_dir / "prompts.py").exists()

    def test_has_requirements(self, hal_dir):
        assert (hal_dir / "requirements.txt").exists()

    def test_requirements_has_smolagents(self, hal_dir):
        reqs = (hal_dir / "requirements.txt").read_text()
        assert "smolagents" in reqs

    def test_agent_py_has_agent_class(self, hal_dir):
        content = (hal_dir / "agent.py").read_text()
        assert "class HALGeneralistAgent" in content
        assert "Agent" in content  # subclasses Agent

    def test_agent_py_has_main(self, hal_dir):
        content = (hal_dir / "agent.py").read_text()
        assert '__name__ == "__main__"' in content


# ── Single-file agents in agents/ ────────────────────────────────────

class TestAgentsCopied:
    """Verify single-file agents exist in agents/."""

    @pytest.fixture
    def agents_dir(self):
        return Path(__file__).parent.parent / "agents"

    def test_echo_agent_exists(self, agents_dir):
        assert (agents_dir / "echo_agent.py").exists()

    def test_simple_qa_agent_exists(self, agents_dir):
        assert (agents_dir / "simple_qa_agent.py").exists()

    def test_metrics_agent_exists(self, agents_dir):
        assert (agents_dir / "metrics_agent.py").exists()

    def test_readme_exists(self, agents_dir):
        assert (agents_dir / "README.md").exists()


# ── CLI integration for agent resolution ──────────────────────────────

class TestCLIAgentResolution:
    """Test that CLI commands accept various agent path formats."""

    def test_run_one_help(self):
        from click.testing import CliRunner
        from harness.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["run-one", "--help"])
        assert result.exit_code == 0
        assert "agent" in result.output.lower()

    def test_run_help(self):
        from click.testing import CliRunner
        from harness.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "agent" in result.output.lower()

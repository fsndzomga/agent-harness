"""Tests for RunRecord and RunRecordStore."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from harness.run_metadata import RunRecord, RunRecordStore, create_run_record


class TestRunRecord:
    """Test RunRecord dataclass."""
    
    def test_create_run_record(self):
        record = create_run_record(
            run_id="run_001",
            benchmark="arithmetic",
            task_id="task_001",
            agent_name="test-agent",
            score=85.0,
            cost_usd=0.05,
            model_costs={"gpt-4o": 0.05},
            agent_metadata={"planning_strategy": "react", "tools": ["search"]},
        )
        
        assert record.run_id == "run_001"
        assert record.benchmark == "arithmetic"
        assert record.task_id == "task_001"
        assert record.agent_name == "test-agent"
        assert record.score == 85.0
        assert record.cost_usd == 0.05
        assert record.model_costs == {"gpt-4o": 0.05}
        assert record.agent_metadata["planning_strategy"] == "react"
        assert isinstance(record.timestamp, datetime)

    def test_create_run_record_defaults(self):
        record = create_run_record(
            run_id="run_002",
            benchmark="gaia",
            task_id="task_002",
            agent_name="test-agent",
            score=50.0,
        )
        
        assert record.cost_usd == 0.0
        assert record.model_costs == {}
        assert record.agent_metadata == {}
        assert isinstance(record.timestamp, datetime)

    def test_to_dict(self):
        record = create_run_record(
            run_id="run_001",
            benchmark="arithmetic",
            task_id="task_001",
            agent_name="test-agent",
            score=85.0,
            agent_metadata={"key": "value"},
        )
        
        d = record.to_dict()
        assert d["run_id"] == "run_001"
        assert d["benchmark"] == "arithmetic"
        assert d["agent_metadata"]["key"] == "value"
        # Timestamp should be ISO string
        assert isinstance(d["timestamp"], str)

    def test_from_dict(self):
        data = {
            "run_id": "run_003",
            "benchmark": "gaia",
            "task_id": "task_003",
            "agent_name": "my-agent",
            "agent_metadata": {"prompt_version": "v2"},
            "score": 90.0,
            "cost_usd": 0.10,
            "model_costs": {"claude-3": 0.10},
            "timestamp": "2026-02-10T12:00:00",
        }
        
        record = RunRecord.from_dict(data)
        assert record.run_id == "run_003"
        assert record.agent_metadata["prompt_version"] == "v2"
        assert isinstance(record.timestamp, datetime)
        assert record.timestamp.year == 2026

    def test_roundtrip(self):
        original = create_run_record(
            run_id="run_rt",
            benchmark="arithmetic",
            task_id="task_rt",
            agent_name="rt-agent",
            score=75.0,
            cost_usd=0.02,
            model_costs={"deepseek": 0.02},
            agent_metadata={"planner": "o3", "executor": "haiku", "max_retries": 5},
        )
        
        d = original.to_dict()
        restored = RunRecord.from_dict(d)
        
        assert restored.run_id == original.run_id
        assert restored.benchmark == original.benchmark
        assert restored.score == original.score
        assert restored.agent_metadata == original.agent_metadata
        assert restored.model_costs == original.model_costs


class TestRunRecordStore:
    """Test RunRecordStore storage and querying."""
    
    def _make_store(self, tmp_path: Path) -> RunRecordStore:
        store_path = tmp_path / "test_runs.jsonl"
        return RunRecordStore(store_path)
    
    def _add_sample_records(self, store: RunRecordStore):
        configs = [
            ("react", "v1.0", "arithmetic", 85.0, 0.03),
            ("react", "v1.0", "gaia", 60.0, 0.05),
            ("cot", "v1.0", "arithmetic", 70.0, 0.02),
            ("cot", "v1.0", "gaia", 50.0, 0.04),
            ("react", "v2.0", "arithmetic", 90.0, 0.03),
            ("react", "v2.0", "gaia", 70.0, 0.06),
        ]
        
        for i, (strategy, version, benchmark, score, cost) in enumerate(configs):
            store.add_record(create_run_record(
                run_id=f"run_{i:03d}",
                benchmark=benchmark,
                task_id=f"task_{i:03d}",
                agent_name="test-agent",
                score=score,
                cost_usd=cost,
                model_costs={"gpt-4o": cost},
                agent_metadata={
                    "planning_strategy": strategy,
                    "prompt_version": version,
                },
            ))
    
    def test_add_and_query_all(self, tmp_path):
        store = self._make_store(tmp_path)
        self._add_sample_records(store)
        
        all_records = store.query()
        assert len(all_records) == 6

    def test_query_by_benchmark(self, tmp_path):
        store = self._make_store(tmp_path)
        self._add_sample_records(store)
        
        arith = store.query(benchmark="arithmetic")
        assert len(arith) == 3
        assert all(r.benchmark == "arithmetic" for r in arith)

    def test_query_by_agent_metadata(self, tmp_path):
        store = self._make_store(tmp_path)
        self._add_sample_records(store)
        
        react_runs = store.query(agent_metadata__planning_strategy="react")
        assert len(react_runs) == 4
        assert all(r.agent_metadata["planning_strategy"] == "react" for r in react_runs)

    def test_query_combined_filters(self, tmp_path):
        store = self._make_store(tmp_path)
        self._add_sample_records(store)
        
        results = store.query(
            benchmark="arithmetic",
            agent_metadata__planning_strategy="react",
        )
        assert len(results) == 2
        assert all(r.benchmark == "arithmetic" for r in results)
        assert all(r.agent_metadata["planning_strategy"] == "react" for r in results)

    def test_query_by_metadata_version(self, tmp_path):
        store = self._make_store(tmp_path)
        self._add_sample_records(store)
        
        v2 = store.query(agent_metadata__prompt_version="v2.0")
        assert len(v2) == 2
        
        avg_score = sum(r.score for r in v2) / len(v2)
        assert avg_score == 80.0  # (90 + 70) / 2

    def test_get_benchmarks(self, tmp_path):
        store = self._make_store(tmp_path)
        self._add_sample_records(store)
        
        benchmarks = store.get_benchmarks()
        assert benchmarks == {"arithmetic", "gaia"}

    def test_get_agents(self, tmp_path):
        store = self._make_store(tmp_path)
        self._add_sample_records(store)
        
        agents = store.get_agents()
        assert agents == {"test-agent"}

    def test_get_metadata_keys(self, tmp_path):
        store = self._make_store(tmp_path)
        self._add_sample_records(store)
        
        keys = store.get_metadata_keys()
        assert "planning_strategy" in keys
        assert "prompt_version" in keys

    def test_stats(self, tmp_path):
        store = self._make_store(tmp_path)
        self._add_sample_records(store)
        
        stats = store.stats()
        assert stats["total_records"] == 6
        assert "arithmetic" in stats["benchmarks"]
        assert "gaia" in stats["benchmarks"]

    def test_persistence(self, tmp_path):
        """Test that records survive store reload."""
        store_path = tmp_path / "persist_test.jsonl"
        
        # Write records
        store1 = RunRecordStore(store_path)
        store1.add_record(create_run_record(
            run_id="persist_001",
            benchmark="arithmetic",
            task_id="t1",
            agent_name="agent-a",
            score=95.0,
            agent_metadata={"key": "value"},
        ))
        
        # Reload from disk
        store2 = RunRecordStore(store_path)
        results = store2.query(run_id="persist_001")
        assert len(results) == 1
        assert results[0].score == 95.0
        assert results[0].agent_metadata["key"] == "value"

    def test_in_memory_store(self):
        """Test store without storage path (in-memory only)."""
        store = RunRecordStore()
        store.add_record(create_run_record(
            run_id="mem_001",
            benchmark="test",
            task_id="t1",
            agent_name="agent",
            score=100.0,
        ))
        
        results = store.query()
        assert len(results) == 1

    def test_empty_query_returns_all(self, tmp_path):
        store = self._make_store(tmp_path)
        self._add_sample_records(store)
        
        assert len(store.query()) == 6

    def test_no_match_returns_empty(self, tmp_path):
        store = self._make_store(tmp_path)
        self._add_sample_records(store)
        
        results = store.query(benchmark="nonexistent")
        assert results == []

    def test_unknown_field_returns_empty(self, tmp_path):
        store = self._make_store(tmp_path)
        self._add_sample_records(store)
        
        results = store.query(nonexistent_field="value")
        assert results == []


class TestSubmissionAgentMetadata:
    """Test agent_metadata in Submission protocol."""
    
    def test_submission_with_agent_metadata(self):
        from harness.protocol import Submission
        
        line = json.dumps({
            "jsonrpc": "2.0",
            "result": {
                "task_id": "test_1",
                "submission": "42",
                "agent_metadata": {
                    "planning_strategy": "react",
                    "prompt_version": "v2",
                },
            },
            "id": 1,
        })
        
        submission = Submission.from_jsonrpc(line)
        assert submission.agent_metadata["planning_strategy"] == "react"
        assert submission.agent_metadata["prompt_version"] == "v2"
    
    def test_submission_without_agent_metadata(self):
        from harness.protocol import Submission
        
        line = json.dumps({
            "jsonrpc": "2.0",
            "result": {
                "task_id": "test_1",
                "submission": "42",
            },
            "id": 1,
        })
        
        submission = Submission.from_jsonrpc(line)
        assert submission.agent_metadata == {}

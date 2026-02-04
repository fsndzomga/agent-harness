"""
GAIA Benchmark - General AI Assistants evaluation.

GAIA is a benchmark proposing real-world questions that require a set of
fundamental abilities such as reasoning, multi-modality handling, web browsing,
and generally tool-use proficiency.

Paper: https://arxiv.org/abs/2311.12983
Dataset: https://huggingface.co/datasets/gaia-benchmark/GAIA
"""

import re
import os
from typing import Any
from pathlib import Path

from ..protocol import Task
from .base import Benchmark, GradeResult
from .graders import grade_with_pipeline, exact_match, normalized_match, numeric_match


class GAIABenchmark(Benchmark):
    """
    GAIA Benchmark for evaluating General AI Assistants.
    
    GAIA tests real-world questions requiring:
    - Reasoning
    - Multi-modality (images, documents)
    - Web browsing
    - Tool use
    
    Levels:
    - Level 1: Simple questions (few steps)
    - Level 2: Medium complexity (5-10 steps)  
    - Level 3: Hard questions (many steps, expert knowledge)
    
    Files are automatically downloaded from HuggingFace Hub.
    """
    
    name = "gaia"
    description = "General AI Assistants benchmark - real-world questions with tools"
    
    def __init__(
        self,
        split: str = "validation",  # "validation" or "test"
        level: int | None = None,   # 1, 2, or 3 (None = all levels)
        cache_dir: str | Path | None = None,
    ):
        self.split = split
        self.level = level
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._tasks: list[Task] | None = None
        self._answers: dict[str, str] = {}
        self._metadata: dict[str, dict] = {}
        self._dataset = None
    
    def _load_dataset(self):
        """Load GAIA dataset from Hugging Face."""
        if self._dataset is not None:
            return
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets is required for GAIA benchmark. "
                "Install it with: pip install datasets"
            )
        
        # GAIA uses specific config names for splits
        # The validation set has answers, test set does not
        config = "2023_all"  # Use the main config
        
        self._dataset = load_dataset(
            "gaia-benchmark/GAIA",
            config,
            split=self.split,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            trust_remote_code=True,
        )
    
    def _download_file(self, file_path: str) -> str | None:
        """
        Download a file from the GAIA dataset on HuggingFace Hub.
        
        Returns the local path to the downloaded file, or None if no file.
        """
        if not file_path:
            return None
        
        try:
            from huggingface_hub import hf_hub_download
            
            local_path = hf_hub_download(
                repo_id="gaia-benchmark/GAIA",
                filename=file_path,
                repo_type="dataset",
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
            return local_path
        except Exception as e:
            print(f"Warning: Failed to download {file_path}: {e}")
            return None
    
    def get_tasks(self) -> list[Task]:
        if self._tasks is not None:
            return self._tasks
        
        self._load_dataset()
        self._tasks = []
        
        for idx, item in enumerate(self._dataset):
            # Filter by level if specified (Level is string in dataset)
            item_level = int(item.get("Level", "1"))
            if self.level is not None and item_level != self.level:
                continue
            
            task_id = item.get("task_id", f"gaia_{idx:04d}")
            question = item["Question"]
            
            # Build task data
            task_data = {
                "question": question,
                "level": item_level,
            }
            
            # Download and include file if present
            if item.get("file_path"):
                local_file = self._download_file(item["file_path"])
                if local_file:
                    task_data["file_name"] = item.get("file_name", "")
                    task_data["file_path"] = local_file  # Local path to downloaded file
                    task_data["original_file_path"] = item["file_path"]  # Original HF path
            
            # Store answer for grading (validation set only)
            if "Final answer" in item and item["Final answer"]:
                self._answers[task_id] = str(item["Final answer"]).strip()
            
            # Store metadata
            self._metadata[task_id] = {
                "level": item_level,
                "annotator_metadata": item.get("Annotator Metadata", {}),
            }
            
            self._tasks.append(Task(id=task_id, data=task_data))
        
        return self._tasks
    
    def grade(self, task_id: str, submission: str) -> GradeResult:
        """
        Grade using GAIA's official grading approach.
        
        GAIA uses exact match after normalization, with special handling for:
        - Numeric answers (tolerance for floating point)
        - Lists (order may not matter for some questions)
        - Case insensitivity
        """
        expected = self._answers.get(task_id, "")
        
        if not expected:
            # Test set - no answer available
            return GradeResult(
                task_id=task_id,
                passed=False,
                score=0.0,
                expected="[hidden - test set]",
                actual=submission,
                method="no_answer",
                details={"reason": "Test set answers not available"},
            )
        
        # Clean submission
        submission_clean = submission.strip()
        expected_clean = expected.strip()
        
        # Try GAIA-style grading pipeline
        # 1. Exact match
        if exact_match(submission_clean, expected_clean):
            return self._grade_result(task_id, True, expected, submission, "exact")
        
        # 2. Normalized match (case insensitive, whitespace normalized)
        if normalized_match(submission_clean, expected_clean):
            return self._grade_result(task_id, True, expected, submission, "normalized")
        
        # 3. Numeric match with tolerance
        if numeric_match(submission_clean, expected_clean, rtol=1e-2, atol=1e-6):
            return self._grade_result(task_id, True, expected, submission, "numeric")
        
        # 4. Check if answer is contained (for verbose responses)
        if self._contains_answer(submission_clean, expected_clean):
            return self._grade_result(task_id, True, expected, submission, "contains")
        
        # Failed all checks
        return self._grade_result(task_id, False, expected, submission, "none")
    
    def _contains_answer(self, submission: str, expected: str) -> bool:
        """Check if the expected answer is contained in submission."""
        sub_lower = submission.lower()
        exp_lower = expected.lower()
        
        # Direct containment
        if exp_lower in sub_lower:
            return True
        
        # Check for formatted answer patterns like "Answer: X" or "The answer is X"
        patterns = [
            rf"(?:answer|result|solution)[:\s]+{re.escape(exp_lower)}",
            rf"{re.escape(exp_lower)}(?:\s|$|[.,!?])",
        ]
        
        for pattern in patterns:
            if re.search(pattern, sub_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _grade_result(
        self,
        task_id: str,
        passed: bool,
        expected: str,
        actual: str,
        method: str,
    ) -> GradeResult:
        """Create a grade result with metadata."""
        return GradeResult(
            task_id=task_id,
            passed=passed,
            score=1.0 if passed else 0.0,
            expected=expected,
            actual=actual,
            method=method,
            details=self._metadata.get(task_id),
        )
    
    def get_num_tasks(self) -> int:
        self._load_dataset()
        if self.level is not None:
            return sum(1 for item in self._dataset if int(item.get("Level", "1")) == self.level)
        return len(self._dataset)


class GAIALevel1Benchmark(GAIABenchmark):
    """GAIA Level 1 - Simple questions."""
    name = "gaia-level1"
    description = "GAIA Level 1 - Simple questions (few steps)"
    
    def __init__(self, **kwargs):
        super().__init__(level=1, **kwargs)


class GAIALevel2Benchmark(GAIABenchmark):
    """GAIA Level 2 - Medium complexity."""
    name = "gaia-level2"
    description = "GAIA Level 2 - Medium complexity (5-10 steps)"
    
    def __init__(self, **kwargs):
        super().__init__(level=2, **kwargs)


class GAIALevel3Benchmark(GAIABenchmark):
    """GAIA Level 3 - Hard questions."""
    name = "gaia-level3"
    description = "GAIA Level 3 - Hard questions (many steps)"
    
    def __init__(self, **kwargs):
        super().__init__(level=3, **kwargs)

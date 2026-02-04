"""Simple arithmetic benchmark for testing the harness."""

import random
from typing import Any

from ..protocol import Task
from .base import Benchmark, GradeResult
from .graders import grade_with_pipeline


class ArithmeticBenchmark(Benchmark):
    """
    Simple arithmetic benchmark for testing.
    
    Generates random arithmetic problems (addition, subtraction, multiplication).
    Useful for testing the harness without needing external datasets.
    """
    
    name = "arithmetic"
    description = "Simple arithmetic problems for testing"
    
    def __init__(self, num_tasks: int = 20, seed: int = 42):
        self.num_tasks = num_tasks
        self.seed = seed
        self._tasks: list[Task] | None = None
        self._answers: dict[str, str] = {}
    
    def get_tasks(self) -> list[Task]:
        if self._tasks is not None:
            return self._tasks
        
        random.seed(self.seed)
        self._tasks = []
        
        ops = [
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("*", lambda a, b: a * b),
        ]
        
        for i in range(self.num_tasks):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            op_sym, op_fn = random.choice(ops)
            
            question = f"What is {a} {op_sym} {b}? Reply with just the number."
            answer = str(op_fn(a, b))
            
            task_id = f"arith_{i:03d}"
            self._tasks.append(Task(
                id=task_id,
                data={"question": question},
            ))
            self._answers[task_id] = answer
        
        return self._tasks
    
    def grade(self, task_id: str, submission: str) -> GradeResult:
        expected = self._answers.get(task_id, "")
        passed, method = grade_with_pipeline(submission, expected)
        
        return GradeResult(
            task_id=task_id,
            passed=passed,
            score=1.0 if passed else 0.0,
            expected=expected,
            actual=submission,
            method=method,
        )
    
    def get_num_tasks(self) -> int:
        return self.num_tasks

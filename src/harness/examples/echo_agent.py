#!/usr/bin/env python3
"""
Echo agent - simplest possible agent for testing.

This agent just echoes back the task data as JSON.
Useful for testing the harness protocol.
"""

import sys
import json


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        if msg.get("method") == "run_task":
            task_id = msg["params"]["task_id"]
            task_data = msg["params"]["task_data"]
            
            # Echo the task data as the answer
            answer = json.dumps(task_data)
            
            response = {
                "jsonrpc": "2.0",
                "result": {"task_id": task_id, "submission": answer},
                "id": msg.get("id"),
            }
            print(json.dumps(response))
            sys.stdout.flush()


if __name__ == "__main__":
    main()

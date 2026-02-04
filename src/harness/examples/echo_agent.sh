#!/bin/bash
# Echo agent in bash - demonstrates language-agnostic protocol
# Requires jq for JSON parsing

read line
task_id=$(echo "$line" | jq -r '.params.task_id')
task_data=$(echo "$line" | jq -c '.params.task_data')

# Echo the task data as the answer
echo "{\"jsonrpc\": \"2.0\", \"result\": {\"task_id\": \"$task_id\", \"submission\": $task_data}, \"id\": 1}"

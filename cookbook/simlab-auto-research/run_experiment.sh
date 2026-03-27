#!/usr/bin/env bash
# run_experiment.sh — Run SimLab tasks against the ConfigurableAgent
# and print the average reward.
#
# Usage: ./run_experiment.sh [task_id ...]
#   If no task IDs are given, runs all tasks from `simlab tasks list`.
#
# Environment variables:
#   AUTO_RESEARCH_ENV   Environment name (default: auto-research-env)
#   MAX_STEPS           Max agent steps per task (default: 30)
#   SYSTEM_PROMPT_PATH  Path to system prompt file (default: system-prompt.md)
#   TASKS_DIR           Path to local task bundle (optional, for custom envs)
#   SIMLAB_AGENT_API_KEY or OPENAI_API_KEY  Required for the agent
#
# Outputs:
#   Per-task reward lines to stdout
#   Final line: avg_reward=<float>

set -euo pipefail

ENV_NAME="${AUTO_RESEARCH_ENV:-auto-research-env}"
AGENT_PATH="simlab_auto_research.configurable_agent:ConfigurableAgent"
MAX_STEPS="${MAX_STEPS:-30}"
TASKS_DIR="${TASKS_DIR:-}"

# Build the tasks-dir flag if set
tasks_dir_flag=""
if [ -n "$TASKS_DIR" ]; then
    tasks_dir_flag="--tasks-dir $TASKS_DIR"
fi

# Collect task IDs from arguments or from `simlab tasks list`
if [ $# -gt 0 ]; then
    TASKS=("$@")
else
    TASKS=()
    while IFS= read -r line; do
        [ -n "$line" ] && TASKS+=("$line")
    done < <(
        # shellcheck disable=SC2086
        simlab tasks list --env "$ENV_NAME" $tasks_dir_flag 2>/dev/null \
            | tail -n +3 \
            | awk '{print $1}' \
            | grep -v '^$'
    )
fi

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "ERROR: No tasks found. Is the environment initialized?" >&2
    exit 1
fi

total_reward=0
task_count=0

for task_id in "${TASKS[@]}"; do
    echo "--- Running task: $task_id ---"

    # Run the task (tasks run auto-starts the env)
    # shellcheck disable=SC2086
    simlab tasks run \
        --env "$ENV_NAME" \
        --task "$task_id" \
        --agent-import-path "$AGENT_PATH" \
        --max-steps "$MAX_STEPS" \
        $tasks_dir_flag \
        2>&1 || true

    # Find the most recent output directory for this task
    run_dir=$(ls -dt output/agent_run_"${task_id}"_* 2>/dev/null | head -1)

    if [ -z "$run_dir" ]; then
        echo "  reward=0.0 (no output directory)"
        task_count=$((task_count + 1))
        continue
    fi

    # Read reward from verifier output
    reward_file="${run_dir}/verifier/reward.json"
    if [ -f "$reward_file" ]; then
        reward=$(python3 -c "import json; print(json.load(open('${reward_file}'))['reward'])")
    else
        reward="0.0"
    fi

    echo "  reward=$reward"
    total_reward=$(python3 -c "print($total_reward + $reward)")
    task_count=$((task_count + 1))
done

if [ "$task_count" -gt 0 ]; then
    avg_reward=$(python3 -c "print(round($total_reward / $task_count, 4))")
else
    avg_reward="0.0"
fi

echo ""
echo "=== Experiment Results ==="
echo "tasks_run=$task_count"
echo "total_reward=$total_reward"
echo "avg_reward=$avg_reward"

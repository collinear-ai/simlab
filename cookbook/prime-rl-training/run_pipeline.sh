#!/usr/bin/env bash
# Full pipeline: SimLab trajectory collection → prime-rl SFT → RL training
#
# Prerequisites:
#   - SimLab installed: pip install "simulationlab[daytona]"
#   - prime CLI installed: pip install prime
#   - API keys exported (see prime-rl-training.md)
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SIMLAB_TEMPLATE:-erp}"
ENV_NAME="${SIMLAB_ENV_NAME:-prime-rl-env}"
TASK_COUNT="${SIMLAB_TASK_COUNT:-10}"
ROLLOUT_COUNT="${SIMLAB_ROLLOUT_COUNT:-3}"
OUTPUT_DIR="${SCRIPT_DIR}/output"
DATASET_DIR="${SCRIPT_DIR}/dataset"
TASKS_DIR="${SCRIPT_DIR}/generated-tasks"
AGENT_MODEL="${SIMLAB_AGENT_MODEL:-gpt-5.2}"

echo "=== Step 1: Create SimLab environment ==="
simlab env init "${ENV_NAME}" --template "${TEMPLATE}"

echo ""
echo "=== Step 2: Generate tasks ==="
simlab tasks-gen init --preset "${TEMPLATE}" 2>/dev/null || true
simlab tasks-gen run 2>/dev/null || true

echo ""
echo "=== Step 3: List available tasks ==="
TASKS=$(simlab tasks list --env "${ENV_NAME}" --tasks-dir "${TASKS_DIR}" 2>/dev/null | head -20)
echo "${TASKS}"

echo ""
echo "=== Step 4: Run rollouts to collect trajectories ==="
# Get task IDs (first column, skip header lines)
TASK_IDS=$(simlab tasks list --env "${ENV_NAME}" --tasks-dir "${TASKS_DIR}" 2>/dev/null \
  | grep -v "^[-=]" | grep -v "^Task" | awk '{print $1}' | head -"${TASK_COUNT}")

if [ -z "${TASK_IDS}" ]; then
    echo "ERROR: No tasks found. Check your environment and task bundle."
    exit 1
fi

for TASK_ID in ${TASK_IDS}; do
    echo "  Running task: ${TASK_ID}"
    simlab tasks run \
        --env "${ENV_NAME}" \
        --task "${TASK_ID}" \
        --tasks-dir "${TASKS_DIR}" \
        --agent-model "${AGENT_MODEL}" \
        --agent-api-key "${OPENAI_API_KEY:-}" \
        --rollout-count "${ROLLOUT_COUNT}" \
        --max-parallel 2 \
        --daytona \
        2>/dev/null || echo "  Warning: task ${TASK_ID} failed, continuing..."
done

echo ""
echo "=== Step 5: Convert trajectories to SFT dataset ==="
python -m prime_rl_training.collect sft \
    --output-dir "${OUTPUT_DIR}" \
    --save-path "${DATASET_DIR}" \
    --min-reward 0.5 \
    --format jsonl

echo ""
echo "=== Step 6: Build verifiers environment ==="
# Create the environment package for prime-rl
ENV_PKG_DIR="${SCRIPT_DIR}/prime-envs/simlab_tasks"
mkdir -p "${ENV_PKG_DIR}"

# Copy the environment module
cp "${SCRIPT_DIR}/src/prime_rl_training/simlab_env.py" "${ENV_PKG_DIR}/simlab_tasks.py"

cat > "${ENV_PKG_DIR}/pyproject.toml" << 'TOML'
[project]
name = "simlab-tasks"
description = "SimLab task environment for prime-rl training"
tags = ["simlab", "tool-use", "multi-turn", "train", "eval"]
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "verifiers>=0.1.11",
    "datasets",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["simlab_tasks.py", "pyproject.toml"]

[tool.verifiers.eval]
num_examples = 5
rollouts_per_example = 3
TOML

echo "Environment package created at ${ENV_PKG_DIR}"

echo ""
echo "=== Step 7: Push environment to Prime Intellect hub ==="
echo "Run: prime env push simlab-tasks --path ${ENV_PKG_DIR}"
echo "(Skipping automatic push — run manually to review first)"

echo ""
echo "=== Step 8: Run RL training ==="
echo "Run: prime rl run ${SCRIPT_DIR}/configs/rl.toml"
echo "(Skipping automatic training — run manually to review config)"

echo ""
echo "=== Pipeline complete ==="
echo "Dataset saved to: ${DATASET_DIR}"
echo "Environment package: ${ENV_PKG_DIR}"
echo ""
echo "Next steps:"
echo "  1. Review the dataset: head ${DATASET_DIR}/train.jsonl"
echo "  2. Push environment: prime env push simlab-tasks --path ${ENV_PKG_DIR}"
echo "  3. Start RL training: prime rl run configs/rl.toml"
echo "  4. Monitor: prime rl logs <run-id> -f"

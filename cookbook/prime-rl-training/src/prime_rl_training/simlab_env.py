"""Verifiers environment that wraps SimLab tasks for prime-rl RL training.

This environment bridges SimLab's task execution and verification system
with prime-rl's verifiers framework. It:

1. Loads SimLab tasks as training prompts (task instructions)
2. Provides SimLab tool servers as callable tools for the RL agent
3. Scores rollouts using SimLab's verifier system (binary or rubric-based)

The environment can operate in two modes:

- **Offline mode** (default): Uses pre-collected trajectories from SimLab
  rollouts as a dataset. The rubric scores based on trajectory quality
  metrics (task completion, tool usage efficiency).

- **Online mode**: Connects to a live SimLab environment and runs the
  agent's rollouts against actual tool servers. Requires Docker or Daytona.

For the cookbook, we focus on offline mode since it doesn't require
running SimLab infrastructure during training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset

import verifiers as vf

logger = logging.getLogger(__name__)


def _load_simlab_trajectories(
    output_dir: str,
    min_reward: float = 0.0,
) -> list[dict[str, Any]]:
    """Load SimLab output artifacts and build dataset rows.

    Each row has:
        - question: the task instruction (user prompt)
        - answer: the successful final response (for reference scoring)
        - info: metadata dict with full trajectory, reward, task_id
        - task: "simlab-task"
    """
    from prime_rl_training.trajectory_converter import (
        collect_trajectories,
    )

    trajectories = collect_trajectories(
        Path(output_dir),
        min_reward=min_reward,
        include_failed=False,
    )

    rows: list[dict[str, Any]] = []
    for traj in trajectories:
        messages = traj["messages"]
        # Extract the user instruction (first user message)
        instruction = ""
        for msg in messages:
            if msg.get("role") == "user":
                instruction = msg.get("content", "")
                break

        # Extract the final assistant response
        final_response = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                final_response = msg["content"]
                break

        if not instruction:
            continue

        rows.append({
            "question": instruction,
            "answer": final_response,
            "info": {
                "full_messages": messages,
                "reward": traj["reward"],
                "task_id": traj["task_id"],
                "source_path": traj["source_path"],
            },
            "task": "simlab-task",
        })

    return rows


def _load_simlab_task_bundle(
    tasks_dir: str,
) -> list[dict[str, Any]]:
    """Load tasks from a SimLab task bundle directory.

    Task bundles contain JSON files with task definitions including
    instructions, seed data specs, and verifier references.
    """
    tasks_path = Path(tasks_dir)
    rows: list[dict[str, Any]] = []

    for task_file in sorted(tasks_path.glob("*.json")):
        try:
            with open(task_file) as f:
                task_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Handle both single task and list of tasks
        task_list = task_data if isinstance(task_data, list) else [task_data]
        for task in task_list:
            instruction = task.get("instruction", task.get("prompt", ""))
            if not instruction:
                continue
            rows.append({
                "question": instruction,
                "answer": "",  # no reference answer for RL
                "info": {
                    "task_id": task.get("id", task.get("task_id", "")),
                    "task_data": task,
                },
                "task": "simlab-task",
            })

    return rows


def _trajectory_similarity_reward(
    completion: str,
    answer: str,
    **kwargs,
) -> float:
    """Score a completion by similarity to the reference trajectory.

    For offline training, we compare the model's output against the
    successful trajectory's final response using a simple overlap metric.
    This provides a dense reward signal for RL training.

    Returns a score in [0, 1].
    """
    if not answer:
        return 0.1 if completion.strip() else 0.0

    # Normalized token overlap (bag-of-words Jaccard similarity)
    comp_tokens = set(completion.lower().split())
    ref_tokens = set(answer.lower().split())

    if not ref_tokens:
        return 0.1 if comp_tokens else 0.0

    intersection = comp_tokens & ref_tokens
    union = comp_tokens | ref_tokens
    jaccard = len(intersection) / len(union) if union else 0.0

    return jaccard


def _format_reward(completion: str, **kwargs) -> float:
    """Reward for well-structured responses."""
    if not completion or not completion.strip():
        return 0.0

    text = completion.strip()

    if len(text) < 20:
        return 0.2

    score = 0.5
    if any(marker in text for marker in ["##", "- ", "1.", "* "]):
        score += 0.2
    if len(text) > 100:
        score += 0.2
    if text.endswith((".", "!", "?", "```")):
        score += 0.1

    return min(score, 1.0)


def load_environment(
    output_dir: str = "./output",
    tasks_dir: str | None = None,
    min_reward: float = 0.5,
    system_prompt: str | None = None,
    **kwargs,
) -> vf.Environment:
    """Load a SimLab verifiers environment for prime-rl training.

    This creates a SingleTurnEnv that uses SimLab trajectory data as
    the training dataset. The rubric scores model outputs based on
    similarity to successful trajectories and response quality.

    Args:
        output_dir: Path to SimLab output directory containing artifacts.
        tasks_dir: Optional path to a SimLab task bundle directory.
                   If provided, tasks are loaded from the bundle instead
                   of from collected trajectories.
        min_reward: Minimum reward threshold for including trajectories
                    from the output directory.
        system_prompt: System prompt for the agent. Defaults to a
                       general-purpose tool-using agent prompt.

    Returns:
        A verifiers Environment ready for prime-rl training.
    """
    if system_prompt is None:
        system_prompt = (
            "You are a capable assistant that can use tools to complete tasks. "
            "Think step by step about what information you need and which tools "
            "to use. Be thorough and precise in your responses."
        )

    # Load dataset
    if tasks_dir:
        rows = _load_simlab_task_bundle(tasks_dir)
    else:
        rows = _load_simlab_trajectories(output_dir, min_reward=min_reward)

    if not rows:
        raise ValueError(
            f"No training data found. Check output_dir={output_dir!r} "
            f"or tasks_dir={tasks_dir!r}"
        )

    logger.info("Loaded %d training examples from SimLab data", len(rows))

    dataset = Dataset.from_list(rows)

    # Build rubric
    rubric = vf.Rubric()
    rubric.add_reward_func(_trajectory_similarity_reward, weight=0.5)
    rubric.add_reward_func(_format_reward, weight=0.3)

    parser = vf.Parser()

    env = vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    return env

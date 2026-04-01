"""Convert SimLab rollout artifacts into prime-rl compatible datasets.

SimLab artifacts.json contains the full agent trajectory: system prompt,
user instructions, tool calls, tool results, and final observations.
This module converts those into the HuggingFace messages format that
prime-rl expects for SFT training, and optionally into prompt/completion
pairs for simpler setups.

Prime-RL SFT format (messages):
    Each row has a "messages" column containing a list of dicts:
        [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]
    Only assistant turns contribute to loss by default.

Prime-RL SFT format (prompt/completion):
    Each row has "prompt" (user instruction) and "completion" (assistant response).

For tool-calling trajectories, tool calls and tool results are interleaved
as assistant/tool message pairs, matching the OpenAI chat format that
prime-rl tokenizes via chat templates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_artifacts(artifacts_path: Path) -> dict[str, Any]:
    """Load a SimLab artifacts.json file."""
    with open(artifacts_path) as f:
        return json.load(f)


def load_reward(verifier_dir: Path) -> float:
    """Load the reward from a SimLab verifier output directory.

    Looks for reward.json first (structured), falls back to reward.txt (plain).
    Returns 0.0 if no reward file is found.
    """
    reward_json = verifier_dir / "reward.json"
    reward_txt = verifier_dir / "reward.txt"

    if reward_json.exists():
        with open(reward_json) as f:
            data = json.load(f)
        # reward.json may have "reward" or "score" key
        return float(data.get("reward", data.get("score", 0.0)))
    elif reward_txt.exists():
        return float(reward_txt.read_text().strip())
    return 0.0


def artifacts_to_messages(artifacts: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a SimLab artifacts dict into a list of chat messages.

    Handles three data shapes commonly found in SimLab artifacts:
    1. "messages" key - already in message format (from reference agent or adapters)
    2. "tool_calls" / "tool_results" lists - interleaved tool-use trajectory
    3. "instruction" + "final_observation" - minimal prompt/response pair

    Returns a list of {"role": ..., "content": ...} dicts.
    """
    messages: list[dict[str, Any]] = []

    # If artifacts already contain a messages list, use it directly
    if "messages" in artifacts and isinstance(artifacts["messages"], list):
        for msg in artifacts["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Normalize tool_calls in assistant messages
            if role == "assistant" and "tool_calls" in msg:
                # Include both text content and tool call info
                tool_calls = msg["tool_calls"]
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                # Add corresponding tool results if present
                for tc in tool_calls:
                    call_id = tc.get("id", "")
                    # Look for a following tool message with matching id
                    # (handled below if tool messages follow in the list)
            elif role == "tool":
                messages.append({
                    "role": "tool",
                    "content": content,
                    "tool_call_id": msg.get("tool_call_id", ""),
                })
            else:
                messages.append({"role": role, "content": content})
        return messages

    # Build from structured fields
    instruction = artifacts.get("instruction", "")
    if instruction:
        messages.append({"role": "user", "content": instruction})

    # Interleave tool calls and results chronologically
    tool_calls = artifacts.get("tool_calls", [])
    tool_results = artifacts.get("tool_results", [])

    for i, tc in enumerate(tool_calls):
        # Assistant makes a tool call
        tool_name = tc.get("name", tc.get("tool_name", "unknown"))
        tool_args = tc.get("arguments", tc.get("args", tc.get("input", {})))
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                pass

        call_id = tc.get("id", f"call_{i}")
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args),
                },
            }],
        })

        # Corresponding tool result
        if i < len(tool_results):
            result = tool_results[i]
            result_content = result.get("content", result.get("output", result.get("result", "")))
            if not isinstance(result_content, str):
                result_content = json.dumps(result_content)
            messages.append({
                "role": "tool",
                "content": result_content,
                "tool_call_id": call_id,
            })

    # Final assistant response
    final = artifacts.get("final_observation", artifacts.get("final_output", ""))
    if final:
        messages.append({"role": "assistant", "content": final})

    return messages


def collect_trajectories(
    output_dir: Path,
    *,
    min_reward: float = 0.0,
    include_failed: bool = False,
) -> list[dict[str, Any]]:
    """Scan a SimLab output directory tree and collect trajectory data.

    Looks for the standard SimLab output structure:
        output_dir/
            agent_run_<task>_<ts>/
                artifacts.json
                verifier/
                    reward.json | reward.txt
            parallel_run_<task>_<ts>/
                rollout_0/
                    artifacts.json
                    verifier/reward.json
                rollout_1/
                    ...

    Args:
        output_dir: Root output directory to scan.
        min_reward: Minimum reward threshold. Trajectories below this are skipped
                    unless include_failed is True.
        include_failed: If True, include all trajectories regardless of reward.

    Returns:
        List of dicts with keys: messages, reward, task_id, source_path.
    """
    output_dir = Path(output_dir)
    trajectories: list[dict[str, Any]] = []

    # Find all artifacts.json files
    for artifacts_path in sorted(output_dir.rglob("artifacts.json")):
        verifier_dir = artifacts_path.parent / "verifier"
        reward = load_reward(verifier_dir)

        if not include_failed and reward < min_reward:
            logger.debug("Skipping %s (reward=%.2f < %.2f)", artifacts_path, reward, min_reward)
            continue

        try:
            artifacts = load_artifacts(artifacts_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load %s: %s", artifacts_path, exc)
            continue

        messages = artifacts_to_messages(artifacts)
        if not messages:
            logger.warning("No messages extracted from %s", artifacts_path)
            continue

        # Try to extract task_id from directory name
        task_id = ""
        dir_name = artifacts_path.parent.name
        parent_name = artifacts_path.parent.parent.name
        for name in (dir_name, parent_name):
            if name.startswith("agent_run_") or name.startswith("parallel_run_"):
                parts = name.split("_")
                # agent_run_<task_id>_<timestamp> or parallel_run_<task_id>_<timestamp>
                if len(parts) >= 3:
                    task_id = "_".join(parts[2:-1])  # everything between prefix and timestamp
                    break

        trajectories.append({
            "messages": messages,
            "reward": reward,
            "task_id": task_id,
            "source_path": str(artifacts_path),
        })

    logger.info("Collected %d trajectories from %s", len(trajectories), output_dir)
    return trajectories


def trajectories_to_sft_dataset(
    trajectories: list[dict[str, Any]],
    *,
    tool_definitions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert collected trajectories into prime-rl SFT dataset rows.

    Each row has:
        - "messages": list of role/content dicts (prime-rl messages format)
        - "tools": optional JSON string of tool definitions (OpenAI format)

    Args:
        trajectories: Output of collect_trajectories().
        tool_definitions: Optional list of tool schemas in OpenAI function format.
                          If provided, added to every row so the model learns tool use.

    Returns:
        List of dataset rows ready for HuggingFace Dataset.from_list().
    """
    rows: list[dict[str, Any]] = []

    for traj in trajectories:
        row: dict[str, Any] = {"messages": traj["messages"]}
        if tool_definitions:
            row["tools"] = json.dumps(tool_definitions)
        rows.append(row)

    return rows


def trajectories_to_prompt_completion(
    trajectories: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Convert trajectories to simple prompt/completion pairs.

    Uses the first user message as prompt and the last assistant message
    as completion. Useful for simpler SFT without tool-use structure.
    """
    rows: list[dict[str, str]] = []

    for traj in trajectories:
        messages = traj["messages"]
        prompt = ""
        completion = ""
        for msg in messages:
            if msg["role"] == "user" and not prompt:
                prompt = msg["content"]
            if msg["role"] == "assistant" and msg.get("content"):
                completion = msg["content"]
        if prompt and completion:
            rows.append({"prompt": prompt, "completion": completion})

    return rows


def save_dataset(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    format: str = "jsonl",
) -> Path:
    """Save dataset rows to disk.

    Args:
        rows: Dataset rows (from trajectories_to_sft_dataset or
              trajectories_to_prompt_completion).
        output_path: Directory to save into.
        format: "jsonl" for JSON Lines, "parquet" for Parquet (requires datasets lib).

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        from datasets import Dataset

        ds = Dataset.from_list(rows)
        file_path = output_path / "train.parquet"
        ds.to_parquet(str(file_path))
        logger.info("Saved %d rows to %s", len(rows), file_path)
        return file_path

    # Default: JSONL
    file_path = output_path / "train.jsonl"
    with open(file_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Saved %d rows to %s", len(rows), file_path)
    return file_path


def push_to_hub(
    rows: list[dict[str, Any]],
    repo_id: str,
    *,
    split: str = "train",
    private: bool = True,
) -> str:
    """Push dataset to HuggingFace Hub for prime-rl to consume.

    Args:
        rows: Dataset rows.
        repo_id: HuggingFace repo ID (e.g., "myorg/simlab-sft-data").
        split: Dataset split name.
        private: Whether to create a private repo.

    Returns:
        The repo URL.
    """
    from datasets import Dataset

    ds = Dataset.from_list(rows)
    ds.push_to_hub(repo_id, split=split, private=private)
    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Pushed %d rows to %s", len(rows), url)
    return url

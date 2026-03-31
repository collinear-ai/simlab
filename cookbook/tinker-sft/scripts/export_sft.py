#!/usr/bin/env python3
"""Convert SimLab artifacts to Tinker SFT JSONL.

Reads rollout artifacts from SimLab output directories, filters by reward,
reconstructs full OpenAI-format messages (recovering tool observations from
the tool_results array), and writes a JSONL file suitable for Tinker SFT.

Usage:
    python export_sft.py output/ training_data.jsonl        # successful only
    python export_sft.py output/ training_data.jsonl 0.0    # all rollouts
"""

import json
import sys
from pathlib import Path


def convert_artifacts(output_dir: str, out_path: str, min_reward: float = 1.0) -> None:
    output = Path(output_dir)
    trajectories = []

    for rollout_dir in sorted(output.rglob("rollout_*")):
        artifacts_file = rollout_dir / "artifacts.json"
        reward_file = rollout_dir / "verifier" / "reward.json"
        if not artifacts_file.exists():
            continue

        # Filter by reward if reward file exists
        if reward_file.exists():
            reward_data = json.loads(reward_file.read_text())
            if reward_data.get("reward", 0.0) < min_reward:
                continue
        elif min_reward > 0.0:
            # No reward file and filtering is on — skip
            continue

        artifacts = json.loads(artifacts_file.read_text())
        messages = artifacts.get("messages", [])
        tool_results = artifacts.get("tool_results", [])

        converted = []
        tool_result_idx = 0

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "assistant" and isinstance(content, dict):
                # Assistant message with tool calls
                text = content.get("content", "") or ""
                raw_calls = content.get("tool_calls", [])
                out_msg = {"role": "assistant", "content": text}
                if raw_calls:
                    out_msg["tool_calls"] = raw_calls
                converted.append(out_msg)

            elif role == "tool" and isinstance(content, dict):
                # Tool result — recover full observation from tool_results array
                tool_call_id = content.get("tool_call_id", "")
                tool_name = content.get("tool_name", content.get("name", ""))
                if tool_result_idx < len(tool_results):
                    obs = tool_results[tool_result_idx]["observation"]
                    full_content = json.dumps(obs) if not isinstance(obs, str) else obs
                    tool_result_idx += 1
                else:
                    full_content = content.get("summary", "")
                converted.append({
                    "role": "tool",
                    "content": full_content,
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                })

            else:
                # User or plain assistant message
                converted.append({
                    "role": role,
                    "content": content if isinstance(content, str) else json.dumps(content),
                })

        if converted:
            trajectories.append({"messages": converted})

    Path(out_path).write_text(
        "\n".join(json.dumps(t) for t in trajectories) + "\n"
    )
    print(f"Wrote {len(trajectories)} trajectories to {out_path}")


if __name__ == "__main__":
    convert_artifacts(
        output_dir=sys.argv[1] if len(sys.argv) > 1 else "output",
        out_path=sys.argv[2] if len(sys.argv) > 2 else "training_data.jsonl",
        min_reward=float(sys.argv[3]) if len(sys.argv) > 3 else 1.0,
    )

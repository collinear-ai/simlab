#!/usr/bin/env python3
"""CLI script to collect SimLab trajectories and convert them for prime-rl.

Usage:
    # Collect from SimLab output directory and save as SFT dataset
    python -m prime_rl_training.collect sft --output-dir ./output --save-path ./dataset

    # Collect and push to HuggingFace Hub
    python -m prime_rl_training.collect sft --output-dir ./output --push-to myorg/simlab-sft

    # Build prompt dataset for RL training
    python -m prime_rl_training.collect rl --output-dir ./output --save-path ./rl-dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def cmd_sft(args: argparse.Namespace) -> None:
    """Collect trajectories and convert to SFT dataset."""
    from prime_rl_training.trajectory_converter import (
        collect_trajectories,
        push_to_hub,
        save_dataset,
        trajectories_to_sft_dataset,
    )

    trajectories = collect_trajectories(
        Path(args.output_dir),
        min_reward=args.min_reward,
        include_failed=args.include_failed,
    )

    if not trajectories:
        logger.error("No trajectories found in %s", args.output_dir)
        sys.exit(1)

    logger.info("Collected %d trajectories", len(trajectories))

    # Load tool definitions if provided
    tool_definitions = None
    if args.tools_file:
        with open(args.tools_file) as f:
            tool_definitions = json.load(f)

    rows = trajectories_to_sft_dataset(
        trajectories,
        tool_definitions=tool_definitions,
    )

    logger.info("Converted to %d SFT dataset rows", len(rows))

    if args.push_to:
        url = push_to_hub(rows, args.push_to, private=not args.public)
        logger.info("Pushed to %s", url)
    else:
        path = save_dataset(rows, Path(args.save_path), format=args.format)
        logger.info("Saved to %s", path)


def cmd_rl(args: argparse.Namespace) -> None:
    """Collect trajectories and build RL prompt dataset."""
    from prime_rl_training.trajectory_converter import collect_trajectories

    trajectories = collect_trajectories(
        Path(args.output_dir),
        min_reward=args.min_reward,
        include_failed=False,
    )

    if not trajectories:
        logger.error("No trajectories found in %s", args.output_dir)
        sys.exit(1)

    logger.info("Collected %d trajectories", len(trajectories))

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Build prompt dataset (question/answer/info/task format for verifiers)
    rows = []
    for i, traj in enumerate(trajectories):
        prompt = ""
        answer = ""
        for msg in traj["messages"]:
            if msg.get("role") == "user" and not prompt:
                prompt = msg["content"]
            if msg.get("role") == "assistant" and msg.get("content"):
                answer = msg["content"]

        if not prompt:
            continue

        rows.append({
            "question": prompt,
            "answer": answer,
            "info": json.dumps({
                "reward": traj.get("reward", 0.0),
                "task_id": traj.get("task_id", f"simlab_{i}"),
            }),
            "task": "simlab-task",
        })

    file_path = save_path / "train.jsonl"
    with open(file_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    logger.info("Saved %d RL prompt rows to %s", len(rows), file_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect SimLab trajectories for prime-rl training"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # SFT subcommand
    sft_parser = subparsers.add_parser("sft", help="Build SFT dataset from trajectories")
    sft_parser.add_argument("--output-dir", required=True, help="SimLab output directory")
    sft_parser.add_argument("--save-path", default="./dataset", help="Where to save the dataset")
    sft_parser.add_argument("--push-to", help="HuggingFace repo ID to push to")
    sft_parser.add_argument("--public", action="store_true", help="Make HF repo public")
    sft_parser.add_argument("--format", choices=["jsonl", "parquet"], default="jsonl")
    sft_parser.add_argument("--min-reward", type=float, default=0.5, help="Min reward threshold")
    sft_parser.add_argument("--include-failed", action="store_true", help="Include failed trajectories")
    sft_parser.add_argument("--tools-file", help="JSON file with tool definitions")

    # RL subcommand
    rl_parser = subparsers.add_parser("rl", help="Build RL prompt dataset")
    rl_parser.add_argument("--output-dir", required=True, help="SimLab output directory")
    rl_parser.add_argument("--save-path", default="./rl-dataset", help="Where to save")
    rl_parser.add_argument("--min-reward", type=float, default=0.5, help="Min reward threshold")

    args = parser.parse_args()

    if args.command == "sft":
        cmd_sft(args)
    elif args.command == "rl":
        cmd_rl(args)


if __name__ == "__main__":
    main()

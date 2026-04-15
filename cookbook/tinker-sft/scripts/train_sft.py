#!/usr/bin/env python3
"""Fine-tune a model on SimLab expert trajectories via Tinker SFT.

Loads a JSONL file of OpenAI-format conversations (from export_sft.py),
deserializes tool calls into pydantic objects, tokenizes via the appropriate
renderer, and runs LoRA fine-tuning with batching and gradient accumulation.

When --group-by-task is set (default), batches are constructed so that all
rollouts of the same task appear in the same batch. With --batch-size 32 and
2 rollouts per task, each batch contains 16 unique tasks x 2 rollouts = 32
datums.

Usage:
    python train_sft.py                                     # uses defaults
    python train_sft.py --model Qwen/Qwen3-8B --renderer qwen3
    python train_sft.py --batch-size 32 --grad-accum 4 --group-by-task
    python train_sft.py --save-name my-checkpoint

Environment:
    TINKER_API_KEY must be set.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import tinker
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.supervised.types import get_tokenizer

if TYPE_CHECKING:
    from tinker_cookbook.renderers import Renderer


def load_datums(
    data_path: str,
    renderer: Renderer,
    max_length: int,
) -> tuple[dict[str, list], list]:
    """Load JSONL and convert to training datums, preserving task grouping."""
    task_groups = defaultdict(list)
    ungrouped = []
    skipped = 0

    with Path(data_path).open() as fh:
        lines = fh.readlines()
    for line in lines:
        traj = json.loads(line)
        msgs = traj["messages"]
        task_id = traj.get("task_id")

        for msg in msgs:
            # Tool calls are serialized inline into content by export_sft.py;
            # strip any stray tool_calls field so it doesn't confuse the renderer.
            msg.pop("tool_calls", None)

        try:
            d = conversation_to_datum(
                msgs, renderer, max_length, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
            )
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Warning: skipped trajectory ({e})")
            continue

        if task_id:
            task_groups[task_id].append(d)
        else:
            ungrouped.append(d)

    if skipped > 3:
        print(f"  ... {skipped} trajectories skipped total")

    return task_groups, ungrouped


def build_task_batches(
    task_groups: dict[str, list],
    ungrouped: list,
    batch_size: int,
    shuffle: bool,
) -> list[list]:
    """Build batches where all rollouts of the same task stay together."""
    task_ids = list(task_groups.keys())
    if shuffle:
        random.shuffle(task_ids)

    batches = []
    current_batch: list = []

    for task_id in task_ids:
        group = task_groups[task_id]
        # If adding this group would exceed batch size, flush current batch
        if current_batch and len(current_batch) + len(group) > batch_size:
            batches.append(current_batch)
            current_batch = []
        current_batch.extend(group)
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []

    # Add ungrouped datums
    for d in ungrouped:
        current_batch.append(d)
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []

    if current_batch:
        batches.append(current_batch)

    return batches


def build_flat_batches(all_datums: list, batch_size: int, shuffle: bool) -> list[list]:
    """Build batches without task grouping."""
    if shuffle:
        random.shuffle(all_datums)
    return [all_datums[i : i + batch_size] for i in range(0, len(all_datums), batch_size)]


def main() -> None:
    """Run Tinker SFT training on SimLab trajectories."""
    parser = argparse.ArgumentParser(description="Tinker SFT on SimLab trajectories")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Student model name")
    parser.add_argument("--renderer", default="qwen3_instruct", help="Renderer name")
    parser.add_argument("--data", default="training_data.jsonl", help="Path to JSONL")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=16384, help="Max sequence length")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0 = all at once)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--save-name", default=None, help="Save checkpoint with this name")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data each epoch")
    parser.add_argument(
        "--group-by-task",
        action="store_true",
        default=True,
        help="Keep rollouts of same task in same batch (default: true)",
    )
    parser.add_argument(
        "--no-group-by-task",
        dest="group_by_task",
        action="store_false",
        help="Disable task grouping",
    )
    args = parser.parse_args()

    # Load tokenizer and renderer
    tokenizer = get_tokenizer(args.model)
    renderer = get_renderer(args.renderer, tokenizer)

    # Load data
    print(f"Loading data from {args.data}...")
    task_groups, ungrouped = load_datums(args.data, renderer, args.max_length)
    total_datums = sum(len(g) for g in task_groups.values()) + len(ungrouped)
    print(f"Converted {total_datums} trajectories from {len(task_groups)} tasks")

    if total_datums == 0:
        print("ERROR: No datums to train on")
        raise SystemExit(1)

    batch_size = args.batch_size if args.batch_size > 0 else total_datums

    if args.group_by_task and task_groups:
        batches = build_task_batches(task_groups, ungrouped, batch_size, args.shuffle)
        rollouts_per_task = total_datums / max(len(task_groups), 1)
        tasks_per_batch = batch_size / max(rollouts_per_task, 1)
        print(
            f"  Grouping by task: ~{rollouts_per_task:.1f} rollouts/task, "
            f"~{tasks_per_batch:.0f} unique tasks per batch of {batch_size}"
        )
    else:
        all_datums = []
        for g in task_groups.values():
            all_datums.extend(g)
        all_datums.extend(ungrouped)
        batches = build_flat_batches(all_datums, batch_size, args.shuffle)

    num_batches = len(batches)
    total_steps = num_batches * args.epochs
    optim_steps = math.ceil(total_steps / args.grad_accum)

    print("Training config:")
    print(f"  Model:           {args.model}")
    print(f"  Datums:          {total_datums}")
    sizes = [len(b) for b in batches[:5]]
    sfx = "..." if num_batches > 5 else ""
    print(f"  Batches:         {num_batches} (sizes: {sizes}{sfx})")
    print(f"  Grad accum:      {args.grad_accum}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Total fwd/bwd:   {total_steps}")
    print(f"  Optimizer steps: {optim_steps}")
    print()

    # Create training client
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.model, rank=args.lora_rank)
    print("Training client created")

    global_step = 0
    accum_count = 0

    for epoch in range(args.epochs):
        if args.shuffle:
            if args.group_by_task and task_groups:
                batches = build_task_batches(task_groups, ungrouped, batch_size, shuffle=True)
            else:
                all_datums = []
                for g in task_groups.values():
                    all_datums.extend(g)
                all_datums.extend(ungrouped)
                batches = build_flat_batches(all_datums, batch_size, shuffle=True)

        for batch_idx, batch in enumerate(batches):
            fwd_bwd = tc.forward_backward(batch, loss_fn="cross_entropy")
            result = fwd_bwd.result()
            global_step += 1
            accum_count += 1

            # Compute mean loss across the batch
            batch_losses = []
            for output in result.loss_fn_outputs:
                if isinstance(output, dict) and "elementwise_loss" in output:
                    tokens = output["elementwise_loss"].data
                    if tokens:
                        batch_losses.append(sum(tokens) / len(tokens))
            mean_loss = sum(batch_losses) / len(batch_losses) if batch_losses else float("nan")

            # Per-step train loss (every fwd/bwd, regardless of grad accumulation)
            print(
                f"  step={global_step} epoch={epoch} batch={batch_idx + 1}/{num_batches} "
                f"loss={mean_loss:.4f} bs={len(batch)}",
                flush=True,
            )

            if accum_count >= args.grad_accum:
                optim = tc.optim_step(
                    tinker.AdamParams(
                        learning_rate=args.lr,
                        beta1=0.9,
                        beta2=0.95,
                        eps=1e-8,
                    )
                )
                optim.result()
                accum_count = 0

        # Flush remaining accumulated gradients at epoch end
        if accum_count > 0:
            optim = tc.optim_step(
                tinker.AdamParams(
                    learning_rate=args.lr,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                )
            )
            optim.result()
            print(f"  Epoch {epoch}: final optimizer step (flushed)")
            accum_count = 0

        print(f"Epoch {epoch} complete")

    # Save checkpoint and persist the fully-qualified tinker path so eval can load it.
    if args.save_name:
        print(f"Saving checkpoint as '{args.save_name}'...")
        save_future = tc.save_weights_for_sampler(args.save_name)
        save_result = save_future.result()
        ckpt_path = save_result.path
        print(f"Checkpoint saved. Path: {ckpt_path}")
        # Write to a sidecar file so downstream eval scripts can pick it up.
        Path("checkpoint_path.txt").write_text(ckpt_path + "\n")
        print("Path written to checkpoint_path.txt")
    else:
        print("Training complete! (no checkpoint saved — use --save-name to enable evaluation)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fine-tune a model on SimLab expert trajectories via Tinker SFT.

Loads a JSONL file of OpenAI-format conversations (from export_sft.py),
deserializes tool calls into pydantic objects, tokenizes via the appropriate
renderer, and runs LoRA fine-tuning through the Tinker service.

Usage:
    python train_sft.py                                     # uses defaults
    python train_sft.py --model Qwen/Qwen3-8B --renderer qwen3
    python train_sft.py --data training_data.jsonl --lr 1e-4

Environment:
    TINKER_API_KEY must be set.
"""

import argparse
import json

from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.supervised.types import get_tokenizer
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import TrainOnWhat, ToolCall
import tinker


def main() -> None:
    parser = argparse.ArgumentParser(description="Tinker SFT on SimLab trajectories")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Student model name")
    parser.add_argument("--renderer", default="qwen3_instruct", help="Renderer name (must match model family)")
    parser.add_argument("--data", default="training_data.jsonl", help="Path to JSONL from export_sft.py")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=16384, help="Max sequence length")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    args = parser.parse_args()

    # Load tokenizer and renderer
    tokenizer = get_tokenizer(args.model)
    renderer = get_renderer(args.renderer, tokenizer)

    # Load and convert conversations
    conversations = []
    for line in open(args.data):
        traj = json.loads(line)
        msgs = traj["messages"]
        for msg in msgs:
            if msg.get("tool_calls"):
                msg["tool_calls"] = [ToolCall.model_validate(tc) for tc in msg["tool_calls"]]
            elif "tool_calls" in msg:
                del msg["tool_calls"]
        conversations.append(msgs)

    datums = [
        conversation_to_datum(conv, renderer, max_length=args.max_length,
                              train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
        for conv in conversations
    ]
    print(f"Converted {len(datums)} trajectories to training datums")

    # Train
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.model, rank=args.lora_rank)
    print("Training client created")

    for epoch in range(args.epochs):
        fwd_bwd = tc.forward_backward(datums, loss_fn="cross_entropy")
        fwd_bwd.result()
        print(f"Epoch {epoch}: forward-backward complete")

        optim = tc.optim_step(tinker.AdamParams(
            learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8,
        ))
        optim.result()
        print(f"Epoch {epoch}: optimizer step complete")

    print("Training complete!")


if __name__ == "__main__":
    main()

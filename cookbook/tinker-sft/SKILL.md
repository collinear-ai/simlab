# SFT with Tinker

Fine-tune a model on expert agent trajectories collected from SimLab using Tinker's supervised learning pipeline.

## Prerequisites

Before starting, confirm these are in place:

1. SimLab is installed with Daytona support: `simlab --version`
2. `SIMLAB_COLLINEAR_API_KEY` is set (from platform.collinear.ai)
3. `DAYTONA_API_KEY` is set (from app.daytona.io)
4. An expert model API key is set (e.g., `OPENAI_API_KEY`)
5. `TINKER_API_KEY` is set (check repo root `.env` or export manually)
6. `tinker-cookbook` is installed: `python -c "import tinker_cookbook"`
7. Verifier is configured: `SIMLAB_VERIFIER_MODEL`, `SIMLAB_VERIFIER_PROVIDER`, and `SIMLAB_VERIFIER_API_KEY`

If any are missing, tell the user which env vars to export or packages to install and wait before proceeding.

## Workflow

### 1. Gather inputs

Ask the user:
- What **expert model** and provider should generate the demonstrations? (e.g., `gpt-5.2` via `openai`)
- What **student model** do you want to fine-tune? (e.g., `Qwen/Qwen3-4B`)
- Do you have an existing SimLab environment, or do you need to create one?
- How many rollouts per task? (recommend 10 for a good dataset)

If no environment exists, follow the [agent-baselining SKILL](../agent-baselining/SKILL.md) steps 1-3 to create and start one.

### 2. Collect expert trajectories

Run the expert model across all tasks with parallel rollouts:

```bash
simlab tasks run \
  --env <env_name> \
  --task <task_id_1> <task_id_2> ... \
  --daytona \
  --rollout-count <count> \
  --max-parallel 3 \
  --agent-model <expert_model> \
  --agent-provider <provider> \
  --agent-api-key "$AGENT_API_KEY"
```

If using a local task bundle, add `--tasks-dir <path>`.

Wait for all runs to complete. Check `output/` for the parallel run directories.

### 3. Convert artifacts to Tinker JSONL

Save the following script as `export_sft.py` and run it. It reads all rollout artifacts, filters to successful trajectories, and converts them to Tinker's expected OpenAI-format JSONL.

```python
#!/usr/bin/env python3
"""Convert SimLab artifacts to Tinker SFT JSONL."""

import json
from pathlib import Path


def convert_artifacts(output_dir: str, out_path: str, min_reward: float = 1.0) -> None:
    output = Path(output_dir)
    trajectories = []

    for rollout_dir in sorted(output.rglob("rollout_*")):
        artifacts_file = rollout_dir / "artifacts.json"
        reward_file = rollout_dir / "verifier" / "reward.json"
        if not artifacts_file.exists() or not reward_file.exists():
            continue

        reward_data = json.loads(reward_file.read_text())
        if reward_data.get("reward", 0.0) < min_reward:
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
                # Assistant message with tool calls — reconstruct OpenAI format
                text = content.get("content", "") or ""
                raw_calls = content.get("tool_calls", [])
                openai_calls = []
                for tc in raw_calls:
                    args = tc.get("arguments", {})
                    openai_calls.append({
                        "type": "function",
                        "id": tc.get("id", ""),
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                        },
                    })
                out_msg = {"role": "assistant", "content": text}
                if openai_calls:
                    out_msg["tool_calls"] = openai_calls
                converted.append(out_msg)

            elif role == "tool" and isinstance(content, dict):
                # Tool result — recover full observation from tool_results array
                tool_call_id = content.get("tool_call_id", "")
                tool_name = content.get("tool_name", "")
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
    import sys
    convert_artifacts(
        output_dir=sys.argv[1] if len(sys.argv) > 1 else "output",
        out_path=sys.argv[2] if len(sys.argv) > 2 else "training_data.jsonl",
    )
```

Run:

```bash
python export_sft.py output/ training_data.jsonl
```

Show the user the trajectory count. If zero, check whether any rollouts passed verification.

Verify a sample looks correct:

```bash
head -1 training_data.jsonl | python -m json.tool
```

Confirm assistant messages with tool calls have the `tool_calls` array with `type`, `id`, and `function` fields. Confirm tool messages have `content` with the full observation (not just a summary).

### 4. Configure and run SFT

Create `train_sft.py` with the student model and dataset path. The renderer is chosen automatically based on the model to produce faithful chat template tokens (e.g., `<tool_call>` / `<tool_response>` for Qwen3).

```python
#!/usr/bin/env python3
"""Fine-tune a model on SimLab expert trajectories via Tinker SFT."""

from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig, TrainOnWhat
from tinker_cookbook.renderers import model_info

MODEL_NAME = "<student_model>"               # e.g., "Qwen/Qwen3-4B"
DATA_PATH = "training_data.jsonl"
LOG_PATH = "/tmp/tinker-sft-simlab"

renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)

common_config = ChatDatasetBuilderCommonConfig(
    model_name_for_tokenizer=MODEL_NAME,
    renderer_name=renderer_name,
    max_length=32768,
    batch_size=128,
    train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
)

dataset = FromConversationFileBuilder(
    common_config=common_config,
    file_path=DATA_PATH,
)

config = train.Config(
    log_path=LOG_PATH,
    model_name=MODEL_NAME,
    dataset_builder=dataset,
    learning_rate=2e-4,
    lr_schedule="linear",
    num_epochs=1,
    eval_every=8,
)

if __name__ == "__main__":
    train.train(config)
```

Replace `<student_model>` with the user's chosen model. Key settings to confirm with the user:

- **`max_length=32768`** — Needs to fit full trajectories. If trajectories are short, 16384 may suffice.
- **`learning_rate=2e-4`** — Tinker's recommended default.
- **`train_on_what=ALL_ASSISTANT_MESSAGES`** — Model learns all assistant turns including tool calls; prompt and tool response tokens get zero loss weight.

Run:

```bash
python train_sft.py
```

### 5. Evaluate the fine-tuned model

After training completes, evaluate the fine-tuned model back in SimLab. The user needs a custom agent that calls the Tinker sampling client. See `tinker-sft.md` Step 4 for the full `TinkerSFTAgent` implementation.

Run evaluation:

```bash
simlab tasks run \
  --env <env_name> \
  --task <task_id_1> <task_id_2> ... \
  --daytona \
  --rollout-count 5 \
  --agent-import-path tinker_sft_agent:TinkerSFTAgent
```

### 6. Present results

Collect `summary.json` from both the expert baseline (Step 2) and the fine-tuned model (Step 5).

Present a comparison table:

| Task | Expert Success Rate | Expert Avg Steps | SFT Success Rate | SFT Avg Steps |
|------|-------------------|-----------------|-----------------|--------------|

**Overall stats:**
- Total trajectories used for training (from Step 3)
- Expert overall success rate
- SFT model overall success rate
- Delta (improvement or regression)

If the SFT model underperforms significantly, suggest:
- Collecting more expert demonstrations (increase rollout count)
- Filtering for shorter successful trajectories (more efficient demonstrations)
- Increasing `max_length` if trajectories are being truncated
- Trying a larger student model

### 7. Tear down

```bash
simlab env down <env_name> --daytona
```

## Troubleshooting

- **`simlab: command not found`** — Install with `uv add "simlab[daytona] @ git+https://github.com/collinear-ai/simlab.git"`.
- **`ModuleNotFoundError: tinker_cookbook`** — Install with `uv pip install tinker-cookbook`.
- **`TINKER_API_KEY` not set** — Check repo root `.env` or export manually.
- **Zero trajectories after conversion** — No rollouts passed verification. Lower `min_reward` threshold or check verifier configuration.
- **Training OOM or slow** — Reduce `max_length` or `batch_size`. Check trajectory lengths in the JSONL.
- **Tool calls missing in JSONL** — Verify `artifacts.json` has `tool_results` populated. The reference agent records these; custom agents may not.

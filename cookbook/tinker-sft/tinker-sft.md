# SFT with Tinker

Collect expert agent trajectories from SimLab and fine-tune a smaller model to imitate them using supervised fine-tuning (SFT) on [Tinker](https://tinker-docs.thinkingmachines.ai/). The expert model generates demonstrations across SimLab tasks, successful trajectories are converted to Tinker's conversation format, and the student model learns to reproduce the expert's tool-calling behavior.

## Prerequisites

- **SimLab** installed with Daytona support:
  ```bash
  uv add "simlab[daytona] @ git+https://github.com/collinear-ai/simlab.git"
  ```
- **tinker-cookbook** installed:
  ```bash
  uv pip install tinker-cookbook
  ```
- **API keys** exported:
  ```bash
  export SIMLAB_COLLINEAR_API_KEY="col_..."   # from platform.collinear.ai
  export DAYTONA_API_KEY="dtn_..."            # from app.daytona.io
  export OPENAI_API_KEY="sk-..."              # or your expert model provider's key
  export TINKER_API_KEY="tml-..."             # from Tinker dashboard
  ```
- **Verifier** configured (to score trajectories):
  ```bash
  export SIMLAB_VERIFIER_MODEL="gpt-5.2"
  export SIMLAB_VERIFIER_PROVIDER="openai"
  export SIMLAB_VERIFIER_API_KEY="$OPENAI_API_KEY"
  ```
- An existing SimLab environment with tasks. See [agent-baselining](../agent-baselining/agent-baselining.md) for setup.

## Step 1: Collect expert trajectories

Run a strong expert model across your tasks with multiple rollouts. These successful trajectories become the training demonstrations.

```bash
simlab tasks run \
  --env my-env \
  --task <task_id_1> <task_id_2> <task_id_3> \
  --daytona \
  --rollout-count 10 \
  --max-parallel 3 \
  --agent-model gpt-5.2 \
  --agent-provider openai \
  --agent-api-key "$OPENAI_API_KEY"
```

If using a local task bundle, add `--tasks-dir <path>`.

Output is written to:

```
output/parallel_run_<task_id>_<timestamp>/
  rollout_0/
    artifacts.json          # Full trajectory (messages, tool_calls, tool_results)
    verifier/reward.json    # {"reward": 1.0} or {"reward": 0.0}
  rollout_1/
    ...
  summary.json
```

## Step 2: Convert to Tinker format

SimLab's `artifacts.json` stores messages in a compact internal format. Tinker expects OpenAI-format conversations with proper `tool_calls` structure so that its model-specific renderers (e.g., Qwen3) can produce the correct chat template tokens.

The script below reads all rollout artifacts under `output/`, filters to successful runs (reward = 1.0), reconstructs OpenAI-format messages, and writes a JSONL file.

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
                # Assistant message with tool calls
                text = content.get("content", "") or ""
                raw_calls = content.get("tool_calls", [])
                openai_calls = []
                for tc in raw_calls:
                    if "function" in tc:
                        # Already OpenAI format (newer SimLab versions)
                        openai_calls.append(tc)
                    else:
                        # Legacy format: {"id", "name", "arguments": {...}}
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
                converted.append({"role": role, "content": content if isinstance(content, str) else json.dumps(content)})

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

Save this as `export_sft.py` and run:

```bash
python export_sft.py output/ training_data.jsonl
```

> **Tip:** Inspect a few lines of the output to verify tool calls look correct:
> ```bash
> head -1 training_data.jsonl | python -m json.tool
> ```

## Step 3: Configure and run SFT

Create a training script that uses tinker-cookbook's supervised learning pipeline:

```python
#!/usr/bin/env python3
"""Fine-tune a model on SimLab expert trajectories via Tinker SFT."""

from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig, TrainOnWhat
from tinker_cookbook.renderers import model_info

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-4B"                # Student model to fine-tune
DATA_PATH = "training_data.jsonl"            # From Step 2
LOG_PATH = "/tmp/tinker-sft-simlab"

renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)

common_config = ChatDatasetBuilderCommonConfig(
    model_name_for_tokenizer=MODEL_NAME,
    renderer_name=renderer_name,
    max_length=32768,           # Must accommodate full trajectories
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

Save as `train_sft.py` and run:

```bash
python train_sft.py
```

**Key configuration choices:**

- **`train_on_what=ALL_ASSISTANT_MESSAGES`** — The model learns to produce all assistant turns, including tool calls. Prompt/user/tool-response tokens get zero loss weight.
- **`max_length=32768`** — Agent trajectories with many tool calls can be long. Adjust based on your data; check the conversion output for typical lengths.
- **`learning_rate=2e-4`** — Tinker's recommended starting point. See [SL Hyperparameters](https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams) for tuning guidance.
- **`renderer_name`** — Automatically selected based on model. The Qwen3 renderer produces faithful `<tool_call>` / `<tool_response>` tokens matching the HuggingFace chat template.

## Step 4: Evaluate the fine-tuned model

After training, use Tinker's sampling client to serve the fine-tuned model and evaluate it back in SimLab. Create a custom agent wrapper:

```python
"""Custom SimLab agent that calls a Tinker-hosted fine-tuned model."""

import json
import threading

import tinker

from simlab.agents.base import BaseAgent, BaseEnvironment, RunArtifacts, ToolCall, ToolCallResult


class TinkerSFTAgent(BaseAgent):
    def __init__(self, sampling_client: tinker.SamplingClient):
        self._client = sampling_client

    @staticmethod
    def name() -> str:
        return "tinker-sft-agent"

    def setup(self, environment: BaseEnvironment) -> None:
        pass

    def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: RunArtifacts,
        *,
        stop_event: threading.Event | None = None,
    ) -> None:
        # Build tool schema from environment
        tools = []
        dispatch = {}
        for tool in environment.list_tools():
            server = tool.get("tool_server")
            name = tool.get("name")
            if not server or not name:
                continue
            wire_name = f"{server}__{name}"
            dispatch[wire_name] = (server, name)
            tools.append({
                "type": "function",
                "function": {
                    "name": wire_name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                },
            })

        messages = [{"role": "user", "content": instruction}]
        context.record_message("user", instruction)
        max_steps = context.max_steps or 30

        for _ in range(max_steps):
            if stop_event and stop_event.is_set():
                context.error = "Cancelled"
                return

            response = self._client.chat_completion(
                messages=messages, tools=tools or None,
            )
            message = response.choices[0].message
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                messages.append(message)
                context.record_message("assistant", {
                    "content": message.get("content", ""),
                    "tool_calls": [
                        {"id": tc["id"], "name": tc["function"]["name"],
                         "arguments": json.loads(tc["function"]["arguments"])}
                        for tc in tool_calls
                    ],
                })
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    args = json.loads(tc["function"]["arguments"])
                    if fn_name not in dispatch:
                        result = ToolCallResult(observation=f"Unknown tool: {fn_name}", is_error=True)
                    else:
                        server, actual_name = dispatch[fn_name]
                        result = environment.call_tool(server, actual_name, args)
                    context.record_tool_call(ToolCall(server, actual_name, args), result)
                    content = json.dumps(result.observation) if not isinstance(result.observation, str) else result.observation
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
                continue

            text = message.get("content", "")
            context.record_message("assistant", text)
            context.final_observation = text
            return

        context.error = "Max steps reached"
```

Then evaluate:

```bash
simlab tasks run \
  --env my-env \
  --task <task_id_1> <task_id_2> <task_id_3> \
  --daytona \
  --rollout-count 5 \
  --agent-import-path tinker_sft_agent:TinkerSFTAgent
```

Compare the fine-tuned model's success rate against the expert baseline from Step 1.

## Step 5: Tear down

```bash
simlab env down my-env --daytona
```

## Next steps

- **Increase data quality** — Run more rollouts, use a stronger expert, or filter by step efficiency (prefer shorter successful trajectories).
- **Scale up** — Fine-tune larger student models (e.g., Qwen3-32B) for better performance.
- **RL fine-tuning** — Use the SFT model as a starting point for on-policy RL, where Tinker samples trajectories and SimLab verifiers provide the reward signal.

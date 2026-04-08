# Agent Integrations

SimLab exposes a small framework-neutral agent contract:

- `BaseEnvironment`
  - enumerate available tools
  - call a tool by namespace and name
- `RunArtifacts`
  - record messages, tool calls, tool results, metadata, and final output
- `BaseAgent`
  - execute one instruction against one environment and populate `RunArtifacts`

That contract is intentionally lower-level than any one agent framework. The
reference agent uses LiteLLM directly, while custom integrations can adapt
framework-specific tool and callback APIs onto the same SimLab runtime.

## Adapter Architecture

The shared adapter layer lives under `simlab.agents.adapters`.

Current pieces:

- `simlab.agents.adapters.core`
  - normalized `ToolDescriptor`
  - shared tool wire names such as `email-env__search_emails`
  - dispatch maps and observation stringification
- `simlab.agents.adapters.artifacts`
  - `RunArtifactsRecorder`
  - structural `ToolResultLike` / `ToolEventRecorder` contracts
- `simlab.agents.adapters.langchain`
  - `build_langchain_tools(environment, recorder=...)`
- `simlab.agents.adapters.openai_agents`
  - `build_openai_agents_tools(environment, recorder=...)`
- `simlab.agents.adapters.claude_agent`
  - `build_claude_agent_tools(environment, recorder=...)`

This split keeps the reusable part in core while leaving framework-specific
message formats and orchestration outside the base runtime.

## Shared Adapter Contract

SimLab adapter modules are expected to satisfy a small shared behavioral
contract.

At a minimum, every adapter should:

- expose stable wire names such as `email-env__search_emails`
- build framework-native tools from a `BaseEnvironment`
- record tool calls and tool results through `RunArtifactsRecorder`
- preserve the shared artifact message shape used by SimLab verifiers and
  artifact inspection

That contract is enforced by shared tests under `cli/simlab/tests/`. Framework-
specific adapters can add stricter compatibility checks on top of it when a
backend has tighter schema or invocation requirements.

## Rollout Metrics

SimLab evaluation will show cost, tokens, and duration when the run artifacts
include a `metadata["rollout_metrics"]` payload.

The canonical shape is

- `metadata["rollout_metrics"]["token_usage"]["prompt_tokens_total"]`
- `metadata["rollout_metrics"]["token_usage"]["completion_tokens_total"]`
- `metadata["rollout_metrics"]["timing"]["duration_seconds"]`
- `metadata["rollout_metrics"]["cost"]["estimated_cost_usd"]`
- `metadata["rollout_metrics"]["extensions"]` for adapter-specific metrics

The `simlab.agents.rollout_metrics` helpers are designed to work across
LiteLLM, OpenAI Agents SDK, and LangChain usage payloads by structurally
parsing token usage objects and building the shared rollout schema.

Treat rollout metrics as additive data. Prefer merging into
`metadata["rollout_metrics"]` so other layers can also attach metrics without
overwriting what is already there.

```python
from simlab.agents.rollout_metrics import RolloutMetricsTracker
from simlab.agents.rollout_metrics import Timer


tracker = RolloutMetricsTracker()
run_timer = Timer.start()

# After each model call, record usage from the framework response.
# - LiteLLM response usage is response.usage
# - OpenAI Agents SDK model response usage is model_response.usage
# - LangChain AI message usage is ai_message.usage_metadata
tracker.record_token_usage(getattr(response, "usage", None))

tracker.record_duration_seconds(run_timer.elapsed_seconds())
tracker.merge_into(context.metadata, model="gpt-4o-mini")
```

Cost is an estimate based on LiteLLM's pricing maps. It is not the provider
billed amount.

## Why This Layer Exists

We want SimLab integrations to scale across multiple ecosystems without making
LangChain or any vendor SDK the platform abstraction.

The intended shape is:

- framework-neutral tool and artifact primitives in core
- thin adapter modules for each ecosystem
- application-specific agents and prompts above that layer

That allows future adapters for OpenAI, Anthropic, or Google agent SDKs to
share the same tool discovery, dispatch, and artifact recording substrate.

## Optional Dependencies

SimLab keeps framework SDKs optional.

From the published package:

```bash
uv tool install "simulationlab[langchain]"
uv tool install "simulationlab[openai-agents]"
uv tool install "simulationlab[claude-agents]"
```

From this repo:

```bash
cd cli/simlab
uv sync --dev --extra langchain
uv sync --dev --extra openai-agents
uv sync --dev --extra claude-agents
```

## LangChain / LangGraph Example

The LangChain adapter converts SimLab tools into LangChain `BaseTool`
instances and records tool execution back into `RunArtifacts`.

```python
from simlab.agents import BaseAgent, BaseEnvironment, RunArtifacts
from simlab.agents.adapters import RunArtifactsRecorder
from simlab.agents.adapters.langchain import build_langchain_tools


class MyLangChainAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "my-langchain-agent"

    def setup(self, environment: BaseEnvironment) -> None:
        _ = environment

    def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: RunArtifacts,
    ) -> None:
        recorder = RunArtifactsRecorder(context)
        tools = build_langchain_tools(environment, recorder=recorder)

        # Build your LangChain or LangGraph app here.
        # The app owns its own prompts, state, and model config.
        result = run_my_graph(instruction=instruction, tools=tools, recorder=recorder)
        context.final_observation = result
```

Use this with:

```bash
simlab tasks run --env my-env --task task_id \
  --agent-import-path path.to.agent:MyLangChainAgent
```

## OpenAI Agents SDK Example

The OpenAI Agents adapter converts SimLab tools into OpenAI Agents SDK
`FunctionTool` instances and records tool execution back into `RunArtifacts`.

```python
from simlab.agents import BaseAgent, BaseEnvironment, RunArtifacts
from simlab.agents.adapters import RunArtifactsRecorder
from simlab.agents.adapters.openai_agents import build_openai_agents_tools

from my_app import run_custom_agent


class MyOpenAIAgentsAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "my-openai-agents-agent"

    def setup(self, environment: BaseEnvironment) -> None:
        _ = environment

    def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: RunArtifacts,
    ) -> None:
        recorder = RunArtifactsRecorder(context)
        recorder.on_user_message(instruction)
        tools = build_openai_agents_tools(environment, recorder=recorder)

        result = run_custom_agent(instruction=instruction, tools=tools)
        context.final_observation = str(result.final_output or "")
```

Use this with:

```bash
simlab tasks run --env my-env --task task_id \
  --agent-import-path path.to.agent:MyOpenAIAgentsAgent
```

## Claude Agent SDK Example

The Claude Agent adapter converts SimLab tools into in-process MCP servers
and records tool execution back into `RunArtifacts`.

```python
import asyncio

from simlab.agents import BaseAgent, BaseEnvironment, RunArtifacts
from simlab.agents.adapters import RunArtifactsRecorder
from simlab.agents.adapters.claude_agent import build_claude_agent_tools

from my_app import run_my_claude_agent


class MyClaudeAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "my-claude-agent"

    def setup(self, environment: BaseEnvironment) -> None:
        _ = environment

    def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: RunArtifacts,
    ) -> None:
        recorder = RunArtifactsRecorder(context)
        recorder.on_user_message(instruction)
        mcp_servers, allowed_tools = build_claude_agent_tools(
            environment, recorder=recorder
        )

        result = asyncio.run(
            run_my_claude_agent(
                instruction=instruction,
                mcp_servers=mcp_servers,
                allowed_tools=allowed_tools,
            )
        )
        context.final_observation = str(result or "")
```

Use this with:

```bash
simlab tasks run --env my-env --task task_id \
  --agent-import-path path.to.agent:MyClaudeAgent
```

## Design Guidance

If you are adding a new integration:

- Put SimLab-independent prompts, planners, state machines, and app logic in
  the app or cookbook package.
- Put framework-specific tool conversion and event/callback glue in an adapter
  module.
- Keep the framework-neutral tool descriptor and artifact logic in core.

As a rule:

- If it is about SimLab tools or `RunArtifacts`, it belongs in core.
- If it is about a framework wire format, it belongs in an adapter.
- If it is about business logic, prompts, or graph topology, it belongs in the
  agent implementation.

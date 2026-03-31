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
```

From this repo:

```bash
cd cli/simlab
uv sync --dev --extra langchain
uv sync --dev --extra openai-agents
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

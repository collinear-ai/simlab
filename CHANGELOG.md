# SimLab CLI — Changelog

All notable changes to the SimLab will be documented in this file.

## [Unreleased]

### Added

- Env-local custom tool definitions under `environments/<env>/custom-tools/`
- `simlab env custom-tools add <env> <name>` to scaffold, enable, and
  regenerate a custom tool in one command
- `simlab tools info --env <env>` support for inspecting env-local tools
- Documentation for env-local custom tool workflows and regeneration behavior

### Changed

- Generated env artifacts now detect drift from `env.yaml`,
  `custom-tools/*.yaml`, and `mcp-servers.json`
- Interactive `env up`, `tasks run`, and `tasks seed` now prompt to
  regenerate stale generated files instead of running with outdated compose
  output
- MCP server names are now validated in the environment namespace, including
  conflicts with built-in tools and env-local custom tools
- Daytona build contexts for environment services must stay inside the
  environment bundle

### Fixed

- SimLab no longer runs with silently stale generated env files after manual
  edits to env config inputs

## [0.2.0] — 2026-03-26

### Added:

#### New Toolsets

- **CRM** — Sales pipeline and account management tool server
- **ERP** — Enterprise resource planning tool server with seed data support
- **Web Search** — Internet search powered by Parallel AI
- **Project Management** — Simulated project management tool

#### MCP Support

- **Custom MCP servers at `env init`** — Plug in stdio-based or URL-based MCP
  servers when creating environments
- **MCP env var mapping** — API keys and secrets can be set via
  `SIMLAB_MCP_<SERVER>__<KEY>` overrides
- **Direct MCP tool access** — The reference agent discovers and calls tools
  from configured MCP servers alongside built-in tool servers

#### Parallel Execution

- **Parallel Daytona rollouts** — Run multiple rollouts concurrently via
  ephemeral sandboxes with `--rollout-count` and `--max-parallel`

#### Seeding

- **Seed group channels** — Tasks can provision RocketChat group channels with
  members and seed messages via `seed_group_channels` in task data

#### Agent Integrations

- **LangGraph** — Built-in adapter for connecting LangGraph agents to SimLab
  environments

#### Evaluation & Analysis

- **`simlab eval`** — Post-rollout analysis command for reviewing agent
  performance

#### CLI Improvements

- **Merged `env up` into `tasks run`** — Single command to spin up environment
  and execute tasks
- **Cookbook** — Step-by-step recipes for SimLab use cases (e.g., agent
  baselining, auto-research)

### Changed

- **LiteLLM reference agent** — `ReferenceAgent` is now provider-agnostic via
  LiteLLM
- **Daytona snapshot auto-creation** — `env up --daytona` automatically creates
  the docker-dind snapshot if it doesn't exist
- **Template consolidation** — Unified HR templates (`hr_recruiting`,
  `people_mgmt`) into 1 `hr` template

### Fixed

- Scenario names and descriptions truncated in `templates list`
- Calendar UID lookup failures when seeding generated tasks
- Docker Compose bind mount resolution when source path contains a symlink
- `env down --daytona` losing state file on transient API errors
- Task IDs showing as `…` in `tasks list` table
- Seeding duplicates when re-running `env seed`

### Agent Integrations

- Shared adapter layer — Added `simlab.agents.adapters` as a framework-neutral
  tool descriptor and artifact recording layer for external agent integrations
- LangChain / LangGraph bridge — Added an optional `langchain` dependency extra
  plus `simlab.agents.adapters.langchain.build_langchain_tools(...)` for
  adapting SimLab environments into LangChain tools
- Reduced duplication in agent wiring — The LangGraph cookbook adapter and the
  built-in reference agent now share the same normalized tool descriptor and
  dispatch logic

## [0.1.0] — 2026-03-16

Initial public release of SimLab — a self-serve simulation lab for building
and evaluating long-horizon AI agents.

### Highlights

SimLab lets you spin up isolated sandboxes, run AI agents against realistic
tasks, and evaluate their performance — all from the command line.

### Environment Management

- `simlab env init` — Create environments from scenario templates or
  interactively pick tool servers from the catalog
- `simlab env up / down` — Start and stop environments locally via Docker
  Compose or remotely via Daytona
- `simlab env seed` — Populate running environments with initial data
  (e.g., HRIS records, etc.)
- Custom MCP servers — Plug in your own MCP servers
  at environment init time
- Health-check aware startup — Services wait for dependency health checks
  before starting dependents; startup fails fast when services exit or stall

### Scenario Templates

Four built-in templates to get started quickly:

- **HR Recruiting** — Interview scheduling, offer management, candidate comms
- **People Management** — Onboarding, performance reviews, compensation
- **Coding** — CLI apps, REST APIs, debugging, refactoring via sandboxed bash
- **Customer Support** — Ticket triage, SLA compliance, escalation workflows

Browse with `simlab templates list`.

### Task Execution

- **`simlab tasks list / info / run`** — Browse tasks, inspect instructions,
  and run agents against them
- **Multi-provider agent support** — OpenAI, Fireworks, or any
  litellm-compatible endpoint
- **Configurable tool bundles** — Choose which tool servers the agent can access
- **Local and Daytona execution** — Run tasks on your machine or in remote
  cloud sandboxes

### Task Generation

- **`simlab tasks-gen init`** — Scaffold a task generation config with
  `--preset` support (recruiting, people_mgmt, coding, customer_support)
- **`simlab tasks-gen validate`** — Validate configs locally before submission
- **`simlab tasks-gen run`** — Generate custom tasks via the API with live
  streaming progress

### Evaluation & Verification

- **Programmatic verifiers** — Python-based verifiers that check environment
  state against success criteria, run automatically after each task
- **Reward Model scoring** — Rubric-based evaluation with configurable
  pass thresholds, confidence scores, and per-dimension breakdowns

### Toolset Catalog

- **`simlab tools list / info`** — Browse available tool servers by category
  (core, integrations, specialized)
- Built-in servers: HRIS (Frappe), Email, Calendar, Chat (RocketChat),
  Browser (Playwright), Helpdesk (Frappe), Coding (OpenHands), Google Workspace, SEC Edgar, Twelve Data

# SimLab CLI — Changelog

All notable changes to the SimLab will be documented in this file.

## [Unreleased]

### Environment Management

- MCP gateway for stdio servers — Command-based MCP servers are wired through
  a generated gateway container, while URL-based MCP servers are used directly
- MCP env var mapping — API keys and other command-server secrets can be set in
  the generated env via raw names or `SIMLAB_MCP_<SERVER>__<KEY>` overrides

### Task Execution

- Direct MCP tool access — The reference agent can discover and call tools
  from configured MCP servers alongside built-in SimLab tool servers

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

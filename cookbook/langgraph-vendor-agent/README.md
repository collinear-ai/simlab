# LangGraph Vendor Management Agent

A plan-and-execute LangGraph agent for vendor management tasks, built on SimLab's agent adapter layer.

## Architecture

```
START → planner → executor (react) → reviewer → (loop or compiler) → END
```

- **Planner**: Reads the task and produces a JSON plan of 4-5 steps
- **Executor**: A `create_react_agent` subgraph that handles multi-round tool calling per step
- **Reviewer**: Marks step done, routes back to executor or forward to compiler
- **Compiler**: Combines all step results into the final formatted response

The agent is **domain-agnostic** — the planner reads the task instructions and adapts the plan accordingly. No hardcoded workflows.

## Tools

This cookbook uses two SimLab tool servers:

- **email-env** (port 8040): Email search and reading
- **erp-env** (port 8100): Suppliers, purchase orders, invoices, products, inventory

## Quick Start

```bash
cd cli/simlab/cookbooks/langgraph-vendor-agent
uv sync

# Initialize environment
simlab --environments-dir ./examples env init vendor-mgmt --force --non-interactive

# Start environment (local Docker)
simlab --environments-dir ./examples env up vendor-mgmt

# Or start on Daytona
simlab --environments-dir ./examples env up vendor-mgmt --daytona

# Set model config
export LANGGRAPH_VENDOR_MODEL="mistralai/mistral-small-2603"
export LANGGRAPH_VENDOR_API_KEY="sk-or-..."
export LANGGRAPH_VENDOR_BASE_URL="https://openrouter.ai/api/v1"

# Run a task
simlab --environments-dir ./examples \
  tasks run --env vendor-mgmt \
  --tasks-dir ./examples/task-bundle \
  --task vendor_erp_review \
  --agent-import-path langgraph_vendor_agent.simlab_adapter:VendorManagementAgent \
  --daytona -v

# Stop environment
simlab --environments-dir ./examples env down vendor-mgmt --daytona
```

## Tasks

| Task | Description | Tools Used |
|------|-------------|------------|
| `vendor_erp_review` | Review vendor performance using emails + ERP data | email + ERP |
| `procurement_audit` | Cross-reference emails with ERP to audit procurement issues | email + ERP |

## Running Tests

```bash
uv run pytest tests/ -v
```

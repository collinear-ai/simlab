# SimLab Cookbook

The SimLab Cookbook provides practical recipes designed to help developers explore and get the most out of SimLab. Each recipe is a self-contained walkthrough you can follow manually or hand to your favorite coding agent.

## How recipes are organized

Each recipe lives in its own folder and contains two files:

| File | Audience | Description |
|------|----------|-------------|
| `<recipe>.md` | Humans | Step-by-step guide with explanations, context, and tips. |
| `SKILL.md` | Agents | Structured instructions a coding agent (Claude Code, Codex, etc.) can execute autonomously. |

## Using recipes with an agent

Point your coding agent at the `SKILL.md` inside any recipe folder. For example:

> Read `cookbook/agent-baselining/SKILL.md` and follow the workflow.

The agent will walk through each step, ask you for any required inputs (model, template, API keys), and run the SimLab commands on your behalf.

## Recipes

| Recipe | Description |
|--------|-------------|
| [agent-baselining](agent-baselining/) | Run your agent across a set of tasks to establish baseline performance scores. |
| [langgraph-email-agent](langgraph-email-agent/) | Standalone LangGraph email assistant cookbook with a bundled SimLab environment, local task bundle, and agent `SKILL.md`. |
| [openai-agents-sdk](openai-agents-sdk/) | Customer-style OpenAI Agents SDK cookbook showing how to keep an existing agent app and add a thin SimLab adapter. |
| [secure-agent-eval](secure-agent-eval/) | Evaluate agent behavior through OneCLI's credential proxy — compare correctness, audit for credential leakage, and test rate limit resilience. |
| [simlab-auto-research](simlab-auto-research/) | Autonomous system prompt optimization using the [auto-research](https://github.com/karpathy/autoresearch) pattern. An outer agent iterates on prompts, measured by SimLab task scores. |
| [tinker-sft](tinker-sft/) | Fine-tune a model on expert agent trajectories using SimLab + Tinker SFT. Includes `run.sh` for end-to-end execution. |

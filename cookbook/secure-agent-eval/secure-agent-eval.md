# Secure Agent Evaluation with OneCLI

Evaluate your agent's behavior through OneCLI's credential proxy. Run the same agent on the same tasks twice — once direct, once proxied — then compare correctness, audit for credential leakage, and test rate limit resilience. This gives you evidence that adding a security layer doesn't break your agent and that credentials never leak into trajectories.

## Prerequisites

- **SimLab** installed (Daytona support optional):
  ```bash
  uv tool install simulationlab
  # or with Daytona support:
  uv tool install "simulationlab[daytona]"
  ```
- **API keys** exported:
  ```bash
  export SIMLAB_COLLINEAR_API_KEY="col_..."   # from platform.collinear.ai
  export OPENAI_API_KEY="sk-..."              # or your provider's key
  export TWELVE_DATA_API_KEY="..."            # from twelvedata.com (free tier works)
  ```
- **Verifier** configured (used to score task results):
  ```bash
  export SIMLAB_VERIFIER_MODEL="gpt-5.2"
  export SIMLAB_VERIFIER_PROVIDER="openai"
  export SIMLAB_VERIFIER_API_KEY="$OPENAI_API_KEY"
  ```
- **`curl`** and **`jq`** on PATH
- **Docker** + **Docker Compose**

## Step 1: Create your environment

Pick a pre-built template or build a custom environment.

**Option A — From a template (recommended):**

```bash
# See what's available
simlab templates list

# Create and start an environment from a template
# Use a template that includes tools with external API credentials (e.g., twelve-data)
simlab env init secure-eval --template hr
simlab env up secure-eval
```

> **Tip:** For the most interesting credential-proxy demo, include `twelve-data` in your environment. TwelveData requires a real API key (`TWELVE_DATA_API_KEY`) — giving OneCLI a second credential to vault alongside the LLM key. If your template doesn't include it, use Option B.

**Option B — Custom environment with TwelveData:**

```bash
# Interactive — pick tools from the catalog (select twelve-data + others)
simlab env init secure-eval

# Or bring your own MCP servers
simlab env init secure-eval --mcp-servers ./mcp-servers.json
```

> **Note:** Custom environments (no template) require a local task bundle for listing and running tasks. Generate one with `simlab tasks-gen` or provide an existing directory via `--tasks-dir`. All `tasks list` and `tasks run` commands below must include `--tasks-dir <path>` when using a custom environment.

> See the [QUICKSTART](../QUICKSTART.md) for full details on MCP server configuration and custom environments.

## Step 2: Baseline run (direct)

Run each task individually with the agent calling tool servers and the LLM provider directly. This is the control group. Parallel rollouts require Daytona, which is out of scope for this recipe — run one task at a time.

```bash
simlab tasks list --env secure-eval
```

Pick one or more task IDs, then run each:

```bash
simlab tasks run \
  --env secure-eval \
  --task <task_id> \
  --agent-model <model> \
  --agent-provider <provider> \
  --agent-api-key "$OPENAI_API_KEY"
```

Repeat for each task you want to baseline. Output lands in `output/agent_run_<task_id>_<timestamp>/`. Save the paths — you will compare them against the proxied run later.

## Step 3: Inject OneCLI

Run the setup script to add OneCLI alongside your SimLab environment:

```bash
./cookbook/secure-agent-eval/setup.sh secure-eval
```

The script does the following:

1. Creates a compose override file (`docker-compose.onecli.yml`) alongside SimLab's generated `docker-compose.yml`. This adds the OneCLI gateway, dashboard, and a PostgreSQL database — all on non-conflicting ports.
2. Starts all services together using both compose files.
3. Configures credentials via OneCLI's dashboard API — creates an agent identity, registers your LLM provider key with a host pattern (`api.openai.com`), and optionally registers your TwelveData key (`api.twelvedata.com`). Generates an access token.
4. Runs a smoke test — a curl through the OneCLI gateway to confirm credential injection is active.
5. Prints export commands for your shell.

At the end, the script prints export commands. Run them:

```bash
export HTTPS_PROXY="http://x:<token-from-setup>@localhost:10255"
export PYTHONPATH="./environments/secure-eval/onecli-site${PYTHONPATH:+:$PYTHONPATH}"
```

The `PYTHONPATH` adds a `sitecustomize.py` that disables SSL verification for litellm — needed because OneCLI's MITM proxy uses a self-signed CA that Python's SSL library rejects.

> **Note:** Since OneCLI lives in a separate compose override file, `simlab env init --force` will not affect it. The OneCLI services survive environment re-initialization.

> **Note:** If you use `uv run`, set `UV_ENV_FILE=/dev/null` to prevent `uv` from auto-loading a `.env` file that might override your `OPENAI_API_KEY` with the real value.

## Step 4: Proxied run

With the proxy active, run the same tasks with the same model — but pass a placeholder key instead of your real one:

```bash
HTTPS_PROXY="http://x:<token>@localhost:10255" \
PYTHONPATH="./environments/secure-eval/onecli-site" \
  simlab tasks run \
  --env secure-eval \
  --task <task_id> \
  --agent-model <model> \
  --agent-provider <provider> \
  --agent-api-key "FAKE_KEY"
```

**How the proxy chain works:**

1. The reference agent (LiteLLM/httpx) reads `HTTPS_PROXY` from the host process environment.
2. httpx sends a `CONNECT` request to OneCLI's gateway, authenticating via the access token embedded in the proxy URL.
3. OneCLI identifies the agent and matches the outbound host (`api.openai.com`) against stored credential patterns.
4. OneCLI replaces `Authorization: Bearer FAKE_KEY` with `Authorization: Bearer <real-key>` on the forwarded request.
5. The agent's trajectory only ever contains `FAKE_KEY` — the real key never appears.

Repeat for each task. Save the output paths as "proxied" for comparison.

## Step 5: Compare correctness

Extract reward scores from each run and compare:

```bash
# Baseline
jq '.reward' output/agent_run_<baseline_task_id>_<baseline_timestamp>/verifier/reward.json

# Proxied
jq '.reward' output/agent_run_<proxied_task_id>_<proxied_timestamp>/verifier/reward.json
```

Use this comparison table to track results across tasks:

| Metric | Baseline (Direct) | Proxied (OneCLI) | Delta |
|--------|--------------------|-------------------|-------|
| Success rate | | | |
| Avg steps | | | |
| Failures | | | |

**What to look for:**

- **Identical success rates** — the proxy should not affect correctness.
- **Comparable step counts** — the proxy adds negligible latency relative to LLM response time.
- **New failures in the proxied run** — these indicate proxy-introduced issues (e.g., credential pattern match failures, TLS interception problems).

## Step 6: Credential audit

Scan the **proxied-run** output directories for real API keys. The baseline run is expected to contain them (they were passed directly), so focus on the proxied outputs.

```bash
# Should return zero matches for both keys
grep -r "$OPENAI_API_KEY" output/agent_run_<proxied_task_id>_<proxied_timestamp>/
grep -r "$TWELVE_DATA_API_KEY" output/agent_run_<proxied_task_id>_<proxied_timestamp>/
```

Verify the placeholder key is present (confirming the agent used it):

```bash
# Should find matches
grep -r "FAKE_KEY" output/agent_run_<proxied_task_id>_<proxied_timestamp>/artifacts.json
```

Also check the OneCLI dashboard logs to confirm injection events occurred:

```bash
# Open the dashboard in your browser
open http://localhost:10254
```

**Expected result:** Zero occurrences of real keys in proxied-run artifacts. The placeholder key (`FAKE_KEY`) should appear — that is correct and expected. This proves the proxy is working: your agent never sees real credentials.

## Step 7: Rate limit stress test

Configure an artificially low rate limit via the OneCLI dashboard API to observe how your agent handles throttling:

```bash
curl -X POST http://localhost:10254/api/rules \
  -H "Content-Type: application/json" \
  -d '{
    "name": "LLM rate limit",
    "hostPattern": "api.openai.com",
    "action": "rate_limit",
    "enabled": true,
    "rateLimit": 5,
    "rateLimitWindow": "minute"
  }'
```

Re-run a task through the proxy and observe the agent's behavior:

```bash
HTTPS_PROXY="http://x:<token>@localhost:10255" \
PYTHONPATH="./environments/secure-eval/onecli-site" \
  simlab tasks run \
  --env secure-eval \
  --task <task_id> \
  --agent-model <model> \
  --agent-provider <provider> \
  --agent-api-key "FAKE_KEY"
```

**What to watch:**

- Does the agent retry after getting rate-limited (HTTP 429)?
- Does it fail gracefully with a meaningful error?
- Does it crash or hang?

This tests agent resilience, not OneCLI's functionality. A well-built agent should handle 429s; many don't.

## Step 8: Tear down

Stop all services — both SimLab and OneCLI containers:

```bash
docker compose \
  -f environments/secure-eval/docker-compose.yml \
  -f environments/secure-eval/docker-compose.onecli.yml \
  down
```

Unset the proxy so subsequent commands go direct again:

```bash
unset HTTPS_PROXY
```

## Next steps

- **Vault additional credentials** — If you use custom MCP servers that call external APIs, register those credentials in OneCLI. The setup script already handles TwelveData; use the same pattern for other tool-server keys.
- **Test with a custom agent** — Pass `--agent-import-path path.to.agent:MyAgent` to evaluate your own agent implementation through the proxy.
- **Compare multiple models through the proxy** — Re-run the same tasks with a different `--agent-model` and compare scores side by side, all routed through OneCLI.

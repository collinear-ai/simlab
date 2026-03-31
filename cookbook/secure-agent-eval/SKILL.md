# Secure Agent Evaluation with OneCLI

Run an agent with and without OneCLI's credential proxy to verify correctness, credential isolation, and rate limit resilience.

## Prerequisites

Before starting, confirm these are in place:

1. SimLab is installed: `simlab --version`
2. `SIMLAB_COLLINEAR_API_KEY` is set (from platform.collinear.ai)
3. `OPENAI_API_KEY` is set (or your LLM provider key)
4. `TWELVE_DATA_API_KEY` is set (free tier from twelvedata.com — optional but recommended)
5. Verifier is configured: `SIMLAB_VERIFIER_MODEL`, `SIMLAB_VERIFIER_PROVIDER`, and `SIMLAB_VERIFIER_API_KEY`
6. `curl` and `jq` are available
7. Docker is running: `docker info`

If any are missing, tell the user which env vars to export and wait before proceeding.

## Workflow

### 1. Gather inputs

Ask the user:
- What model and provider do you want to evaluate? (e.g., `gpt-5.2` via `openai`)
- Do you have a preferred template, or want to see what's available?

If the user isn't sure about templates, run:

```bash
simlab templates list
```

Show the output and let them pick.

### 2. Create the environment

```bash
simlab env init secure-eval --template <template>
simlab env up secure-eval
```

If the user wants a custom environment (no template), run `simlab env init secure-eval` without `--template` and follow the interactive prompts. If they have an MCP servers config, use `--mcp-servers <path>`.

**Important:** Custom environments (no template) require a local task bundle. The user must either generate tasks with `simlab tasks-gen` or provide an existing `--tasks-dir` path. All subsequent `tasks list` and `tasks run` commands must include `--tasks-dir <path>`.

### 3. List tasks and select targets

```bash
simlab tasks list --env secure-eval
```

Note the task IDs. Let the user pick which tasks to evaluate.

### 4. Run baseline (direct)

For each selected task:

```bash
simlab tasks run \
  --env secure-eval \
  --task <task_id> \
  --agent-model <model> \
  --agent-provider <provider> \
  --agent-api-key "$OPENAI_API_KEY"
```

Record the output paths (`output/agent_run_<task_id>_<timestamp>/`) as baseline.

### 5. Inject OneCLI

Run the setup script:

```bash
./cookbook/secure-agent-eval/setup.sh secure-eval
```

Parse the printed export commands from the setup output. Set both in your shell:

```bash
export HTTPS_PROXY="http://x:<token-from-setup>@localhost:10255"
export PYTHONPATH="./environments/secure-eval/onecli-site${PYTHONPATH:+:$PYTHONPATH}"
```

The `PYTHONPATH` is required — it loads a `sitecustomize.py` that disables SSL verification for litellm (OneCLI's MITM proxy uses a self-signed CA).

If using `uv run`, also set `UV_ENV_FILE=/dev/null` to prevent `uv` from auto-loading `.env` files that contain the real API key.

If the user's terminal session is separate from your execution environment, tell them to export these themselves and wait for confirmation.

### 6. Run proxied evaluation

For each selected task:

```bash
simlab tasks run \
  --env secure-eval \
  --task <task_id> \
  --agent-model <model> \
  --agent-provider <provider> \
  --agent-api-key "FAKE_KEY"
```

Record output paths as proxied.

### 7. Compare correctness

Read `verifier/reward.json` from each baseline and proxied run:

```bash
jq '.reward' output/agent_run_<baseline_task_id>_<baseline_timestamp>/verifier/reward.json
jq '.reward' output/agent_run_<proxied_task_id>_<proxied_timestamp>/verifier/reward.json
```

Build and present a comparison table:

| Task | Baseline Reward | Proxied Reward | Baseline Steps | Proxied Steps | Match? |
|------|----------------|----------------|----------------|---------------|--------|

Flag any tasks where the proxied run failed but baseline passed — these indicate proxy-introduced issues.

### 8. Credential audit

Scan proxied-run outputs for real API keys:

```bash
grep -r "$OPENAI_API_KEY" output/agent_run_<proxied>_<timestamp>/
grep -r "$TWELVE_DATA_API_KEY" output/agent_run_<proxied>_<timestamp>/
```

Expected: zero matches for both.

Verify the placeholder key IS present:

```bash
grep -r "FAKE_KEY" output/agent_run_<proxied>_<timestamp>/artifacts.json
```

Expected: matches found.

Check OneCLI logs for injection events:

```bash
docker compose \
  -f environments/secure-eval/docker-compose.yml \
  -f environments/secure-eval/docker-compose.onecli.yml \
  logs onecli-app
```

Present pass/fail for the credential audit.

### 9. Rate limit stress test

Create a rate limit via OneCLI dashboard API:

```bash
curl -X POST http://localhost:10254/api/rules \
  -H "Content-Type: application/json" \
  -d '{"name":"LLM rate limit","hostPattern":"api.openai.com","action":"rate_limit","enabled":true,"rateLimit":5,"rateLimitWindow":"minute"}'
```

Re-run one task and observe:
- Does the agent retry?
- Does it fail gracefully?
- Does it crash or hang?

Read the `artifacts.json` to analyze the agent's behavior under throttling.

### 10. Tear down

```bash
docker compose \
  -f environments/secure-eval/docker-compose.yml \
  -f environments/secure-eval/docker-compose.onecli.yml \
  down
unset HTTPS_PROXY
```

### 11. Present results

Show the user a report with:

**Correctness comparison table** (from step 7)

**Credential audit:**
- PASS/FAIL: real key found in proxied artifacts?
- PASS/FAIL: placeholder key present in proxied artifacts?

**Rate limit resilience:**
- How the agent handled throttling
- Whether it retried, failed gracefully, or crashed

This is their secure evaluation report. They can re-run after changing OneCLI rules, models, or providers to measure differences.

## Troubleshooting

- **`setup.sh` fails: "OPENAI_API_KEY must be exported"** — Export your LLM provider key before running setup.
- **OneCLI gateway not reachable** — Check `docker compose ... logs onecli-app`. The gateway may need a moment to start after `setup.sh`.
- **Credential pattern mismatch** — Verify the hostPattern in OneCLI matches your LLM provider URL (e.g., `api.openai.com` for OpenAI).
- **Proxy env vars not working** — `HTTPS_PROXY` must be exported in the host shell where you run `simlab tasks run`, not in the Docker `.env` file.
- **SSL certificate errors** — OneCLI's MITM proxy uses a self-signed CA. Set `PYTHONPATH` to include `environments/<env>/onecli-site/` which has a `sitecustomize.py` that disables SSL verification for litellm.
- **Real key still appears despite FAKE_KEY** — `uv` auto-loads `.env` files. Set `UV_ENV_FILE=/dev/null` before running.
- **`simlab env init --force` and OneCLI** — The OneCLI compose override is a separate file; `--force` only regenerates `docker-compose.yml`. OneCLI services are preserved.
- **No reward files** — The task may not have verifiers defined. Check `simlab tasks list` for tasks with verifiers.

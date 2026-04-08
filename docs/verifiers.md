# Verifiers

Verifiers check whether an agent successfully completed a task and produce a reward score. They run locally after the agent finishes.

We use two verifiers in SimLab:
- **Programmatic Verifiers**: Verifiers that inspect the playground state directly. They compare before/after snapshots of the playground to confirm the agent made the correct changes.
- **Reward Models**: Rubric-based verifiers use reward models to evaluate the agent’s actions against a scoring rubric. The judge reviews the agent’s full trace and assigns a reward score.

## When Verifiers Run

If a task JSON includes a `verifiers` (or `evaluators`) list with `func: python_module` and a `module` path, the CLI runs those verifiers after the agent completes. Tasks without verifiers only produce rollout artifact files — there is no verifier step and no reward files.

## Reward Model Configuration

Configure its credentials via `config.toml`, environment variables, or both:

### config.toml

```toml
[verifier]
model = "gpt-4o-mini"
provider = "openai"
api_key = "sk-..."
```

### Environment Variables

```bash
export SIMLAB_VERIFIER_MODEL="gpt-5.2"
export SIMLAB_VERIFIER_PROVIDER="openai"
export SIMLAB_VERIFIER_API_KEY="$OPENAI_API_KEY"
```

## Verifier Bundle Caching

When using the hosted API (default `https://rl-gym-api.collinear.ai`), the CLI downloads the verifier bundle from the Scenario Manager API on first use and caches it under `environments/<env-name>/verifiers/`.

To clear the cache (e.g. after updating the verifier runtime):

```bash
rm -rf environments/my-env/verifiers
```

The next run re-downloads the current bundle.

## Run Output Layout

### Single Rollout (default)

```
output/agent_run_<task_id>_<timestamp>/
  artifacts.json                        # default rollout artifacts
  agent/trajectory.json                 # ATIF artifacts (when rollout_format = atif)
  verifier/reward.txt                   # "1" or "0"
  verifier/reward.json                  # e.g. {"reward": 1.0}
```

### Parallel Rollouts (`--rollout-count > 1`)

```
output/parallel_run_<task_id>_<timestamp>/
  rollout_0/
    artifacts.json
    verifier/reward.txt
    verifier/reward.json
  rollout_1/
    ...
  summary.json
```

When `rollout_format` is `atif`, each rollout directory writes `agent/trajectory.json` instead of `artifacts.json`.

`summary.json` includes completed/failed counts, average reward, average steps, and per-rollout details.

## ATIF Output Format

To emit an ATIF trajectory for any task run, use one of:

- CLI flag: `--tasks-rollout-format atif`
- `env.yaml`: `rollout_format: atif`
- Config: `[tasks] rollout_format = "atif"`
- Env var: `SIMLAB_TASKS_ROLLOUT_FORMAT=atif`

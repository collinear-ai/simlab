# Troubleshooting

## Clearing Verifier Cache

Verifier bundles are cached under `environments/<env-name>/verifiers/`. If verifier behavior seems stale or you're seeing unexpected results, clear the cache:

```bash
rm -rf environments/my-env/verifiers
```

The next `tasks run` re-downloads the current bundle from the API.

## Using a Local Scenario Manager API

If you need to run against your own Scenario Manager instead of the hosted one (`https://rl-gym-api.collinear.ai`):

```bash
# From the monorepo root: start the API + Postgres only
docker compose up -d postgres scenario-manager-api
export SIMLAB_SCENARIO_MANAGER_API_URL=http://localhost:9011

simlab templates list
simlab tasks list --env my-env
```

The API serves verifier bundles via `GET /scenarios/{scenario_id}/verifiers/bundle`; it needs the monorepo `src/` (mounted in Docker) to read scenario verifier files.

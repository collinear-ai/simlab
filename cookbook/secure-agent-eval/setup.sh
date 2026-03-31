#!/usr/bin/env bash
# setup.sh — Inject OneCLI credential vault into a SimLab environment
#
# Creates a Docker Compose override file (docker-compose.onecli.yml) alongside
# SimLab's generated compose file, then starts everything together. This avoids
# modifying the original compose file — it survives `simlab env init --force`
# and sidesteps YAML injection fragility entirely.
#
# Usage: ./setup.sh <env-name> [--environments-dir <path>]

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
readonly ONECLI_IMAGE="ghcr.io/onecli/onecli:latest"
readonly ONECLI_PG_IMAGE="postgres:17-alpine"
readonly ONECLI_PG_PORT=10256
readonly ONECLI_PG_USER="onecli"
readonly ONECLI_PG_PASS="onecli"
readonly ONECLI_PG_DB="onecli"
readonly ONECLI_DASHBOARD_PORT=10254
readonly ONECLI_GATEWAY_PORT=10255
readonly HEALTH_RETRIES=30
readonly HEALTH_INTERVAL=2

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $(basename "$0") <env-name> [--environments-dir <path>]

Inject OneCLI credential vault into a SimLab environment.

Arguments:
  env-name               Name of the SimLab environment (required)
  --environments-dir     Path to environments directory (default: ./environments)

Environment variables:
  OPENAI_API_KEY         Must be exported before running this script
  TWELVE_DATA_API_KEY    Optional — vaulted alongside the LLM key if set

Example:
  export OPENAI_API_KEY="sk-..."
  export TWELVE_DATA_API_KEY="your-td-key"
  ./setup.sh my-eval-env
EOF
  exit 1
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
ENV_NAME="${1:-}"
if [[ -z "$ENV_NAME" ]]; then
  echo "Error: env-name is required." >&2
  usage
fi
if [[ -n "$ENV_NAME" ]]; then shift; fi

ENVIRONMENTS_DIR="./environments"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --environments-dir)
      ENVIRONMENTS_DIR="${2:-}"
      if [[ -z "$ENVIRONMENTS_DIR" ]]; then
        echo "Error: --environments-dir requires a value." >&2
        usage
      fi
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Error: unknown argument '$1'" >&2
      usage
      ;;
  esac
done

ENV_DIR="${ENVIRONMENTS_DIR}/${ENV_NAME}"
COMPOSE_FILE="${ENV_DIR}/docker-compose.yml"
ONECLI_COMPOSE_FILE="${ENV_DIR}/docker-compose.onecli.yml"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Error: Compose file not found: $COMPOSE_FILE" >&2
  echo "Have you run 'simlab env init ${ENV_NAME}'?" >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Error: OPENAI_API_KEY must be exported." >&2
  exit 1
fi

if [[ -z "${TWELVE_DATA_API_KEY:-}" ]]; then
  echo "Warning: TWELVE_DATA_API_KEY is not set. Only the LLM key will be vaulted." >&2
fi

for cmd in docker curl jq; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "Error: '$cmd' is required but not found on PATH." >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Step 2: Compose override file creation
# ---------------------------------------------------------------------------
create_onecli_compose() {
  if [[ -f "$ONECLI_COMPOSE_FILE" ]]; then
    echo "OneCLI compose override already exists: $ONECLI_COMPOSE_FILE (skipping creation)"
    return 0
  fi

  echo "Creating OneCLI compose override: $ONECLI_COMPOSE_FILE"
  cat > "$ONECLI_COMPOSE_FILE" <<YAML
# OneCLI Credential Vault — compose override for SimLab
# Created by setup.sh. Safe to delete if OneCLI is no longer needed.
# This file is NOT managed by SimLab and survives 'simlab env init --force'.

services:
  onecli-db:
    image: ${ONECLI_PG_IMAGE}
    environment:
      POSTGRES_USER: ${ONECLI_PG_USER}
      POSTGRES_PASSWORD: ${ONECLI_PG_PASS}
      POSTGRES_DB: ${ONECLI_PG_DB}
    ports:
      - "${ONECLI_PG_PORT}:5432"
    volumes:
      - onecli-pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${ONECLI_PG_USER}"]
      interval: 5s
      timeout: 3s
      start_period: 15s
      retries: 10

  onecli-app:
    image: ${ONECLI_IMAGE}
    depends_on:
      onecli-db:
        condition: service_healthy
    environment:
      DATABASE_URL: "postgresql://${ONECLI_PG_USER}:${ONECLI_PG_PASS}@onecli-db:5432/${ONECLI_PG_DB}"
    ports:
      - "${ONECLI_DASHBOARD_PORT}:10254"
      - "${ONECLI_GATEWAY_PORT}:10255"

volumes:
  onecli-pgdata:
YAML
}

# ---------------------------------------------------------------------------
# Step 3: Helpers — dc, wait_for_onecli, configure_credentials
# ---------------------------------------------------------------------------
dc() {
  docker compose -f "$COMPOSE_FILE" -f "$ONECLI_COMPOSE_FILE" "$@"
}

wait_for_onecli() {
  echo "Waiting for OneCLI dashboard to become healthy..."
  local attempt=0
  while [[ $attempt -lt $HEALTH_RETRIES ]]; do
    local code
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${ONECLI_DASHBOARD_PORT}/api/health" 2>/dev/null) || true
    if [[ "$code" == "200" ]]; then
      echo "OneCLI dashboard is ready."
      return 0
    fi
    attempt=$((attempt + 1))
    echo "  Attempt ${attempt}/${HEALTH_RETRIES} — waiting ${HEALTH_INTERVAL}s..."
    sleep "$HEALTH_INTERVAL"
  done

  echo "Error: OneCLI dashboard did not become healthy after $((HEALTH_RETRIES * HEALTH_INTERVAL))s." >&2
  echo "Check logs with: docker compose -f $COMPOSE_FILE -f $ONECLI_COMPOSE_FILE logs onecli-app" >&2
  exit 1
}

configure_credentials() {
  echo "Configuring OneCLI credentials..."

  local api_base="http://localhost:${ONECLI_DASHBOARD_PORT}/api"
  local agent_state_file="${ENV_DIR}/.onecli-agent"

  # Check for existing agent state to avoid creating duplicates
  if [[ -f "$agent_state_file" ]]; then
    local existing_token
    existing_token=$(jq -r '.access_token // empty' "$agent_state_file")
    if [[ -n "$existing_token" ]]; then
      echo "  Agent already configured (loaded from $agent_state_file). Skipping creation."
      AGENT_ACCESS_TOKEN="$existing_token"
      CA_CERT_FILE=$(jq -r '.ca_cert_file // empty' "$agent_state_file")
      SITE_DIR=$(jq -r '.site_dir // empty' "$agent_state_file")
      return 0
    fi
  fi

  # Create agent identity (requires both name and identifier)
  local agent_response
  agent_response=$(curl -s -X POST "${api_base}/agents" \
    -H "Content-Type: application/json" \
    -d '{"name": "SimLab Agent", "identifier": "simlab-agent"}') || true

  local agent_id
  agent_id=$(echo "$agent_response" | jq -r '.id // empty')
  if [[ -z "$agent_id" ]]; then
    echo "Error: Failed to create agent identity." >&2
    echo "Response: $agent_response" >&2
    exit 1
  fi
  echo "  Agent created: id=$agent_id"

  # Access token is returned on the list endpoint, not the create response
  local access_token
  access_token=$(curl -s "${api_base}/agents" | jq -r --arg id "$agent_id" '.[] | select(.id == $id) | .accessToken // empty') || true
  if [[ -z "$access_token" ]]; then
    echo "Error: Could not retrieve access token for agent $agent_id." >&2
    exit 1
  fi

  # Register credential (type: generic, with injectionConfig for header name)
  local secret_response
  secret_response=$(curl -s -X POST "${api_base}/secrets" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
      --arg key "$OPENAI_API_KEY" \
      '{
        name: "openai-api-key",
        type: "generic",
        value: ("Bearer " + $key),
        hostPattern: "api.openai.com",
        injectionConfig: { headerName: "Authorization" }
      }')") || true

  local secret_id
  secret_id=$(echo "$secret_response" | jq -r '.id // empty')
  if [[ -z "$secret_id" ]]; then
    echo "Error: Failed to register credential." >&2
    echo "Response: $secret_response" >&2
    exit 1
  fi
  echo "  Credential registered: id=$secret_id hostPattern=api.openai.com"

  # Register TwelveData credential (if key is available)
  local td_secret_id=""
  if [[ -n "${TWELVE_DATA_API_KEY:-}" ]]; then
    local td_secret_response
    td_secret_response=$(curl -s -X POST "${api_base}/secrets" \
      -H "Content-Type: application/json" \
      -d "$(jq -n \
        --arg key "$TWELVE_DATA_API_KEY" \
        '{
          name: "twelve-data-api-key",
          type: "generic",
          value: $key,
          hostPattern: "api.twelvedata.com",
          injectionConfig: { headerName: "X-API-Key" }
        }')") || true

    td_secret_id=$(echo "$td_secret_response" | jq -r '.id // empty')
    if [[ -n "$td_secret_id" ]]; then
      echo "  Credential registered: id=$td_secret_id hostPattern=api.twelvedata.com"
    else
      echo "  Warning: Failed to register TwelveData credential: $td_secret_response" >&2
    fi
  fi

  # Assign secrets to the agent (selective mode requires explicit assignment)
  local secret_ids
  if [[ -n "$td_secret_id" ]]; then
    secret_ids=$(jq -n --arg s1 "$secret_id" --arg s2 "$td_secret_id" '{secretIds: [$s1, $s2]}')
  else
    secret_ids=$(jq -n --arg sid "$secret_id" '{secretIds: [$sid]}')
  fi

  local assign_response
  assign_response=$(curl -s -X PUT "${api_base}/agents/${agent_id}/secrets" \
    -H "Content-Type: application/json" \
    -d "$secret_ids") || true

  if ! echo "$assign_response" | jq -e '.success' &>/dev/null; then
    echo "Warning: Could not assign secret to agent. Response: $assign_response" >&2
  fi
  echo "  Secret assigned to agent."

  # Download OneCLI CA certificate (needed for HTTPS MITM proxy)
  local ca_cert_file="${ENV_DIR}/onecli-ca.pem"
  local ca_cert
  ca_cert=$(curl -s "${api_base}/container-config" | jq -r '.caCertificate // empty') || true
  if [[ -n "$ca_cert" ]]; then
    echo "$ca_cert" > "$ca_cert_file"
    echo "  CA certificate saved to $ca_cert_file"
  else
    echo "  Warning: Could not retrieve CA certificate. HTTPS proxy may fail with SSL errors." >&2
  fi

  # Create a sitecustomize.py that disables SSL verification for litellm.
  # OneCLI uses MITM HTTPS interception with a self-signed CA. Python's SSL
  # library rejects the cert even with the CA bundle because it's missing the
  # Authority Key Identifier extension. Setting litellm.ssl_verify = False is
  # the pragmatic workaround — the proxy itself is local and trusted.
  local site_dir="${ENV_DIR}/onecli-site"
  mkdir -p "$site_dir"
  cat > "${site_dir}/sitecustomize.py" <<'PYTHON'
try:
    import litellm
    litellm.ssl_verify = False
except ImportError:
    pass
PYTHON
  echo "  SSL workaround installed at ${site_dir}/sitecustomize.py"

  # Persist agent state for idempotent re-runs
  jq -n --arg id "$agent_id" --arg token "$access_token" --arg ca "$ca_cert_file" --arg site "$site_dir" \
    '{agent_id: $id, access_token: $token, ca_cert_file: $ca, site_dir: $site}' > "$agent_state_file"

  # Export for use by smoke_test and print_config
  AGENT_ACCESS_TOKEN="$access_token"
  CA_CERT_FILE="$ca_cert_file"
  SITE_DIR="$site_dir"
}

# ---------------------------------------------------------------------------
# Step 4: Smoke test and output
# ---------------------------------------------------------------------------
smoke_test() {
  echo "Running smoke test (curl through OneCLI gateway)..."

  local proxy_url="http://x:${AGENT_ACCESS_TOKEN}@localhost:${ONECLI_GATEWAY_PORT}"
  local cacert_flag=""
  if [[ -n "${CA_CERT_FILE:-}" && -f "${CA_CERT_FILE}" ]]; then
    cacert_flag="--cacert ${CA_CERT_FILE}"
  else
    cacert_flag="--proxy-insecure"
  fi

  local status_code
  # shellcheck disable=SC2086
  status_code=$(curl -s -o /dev/null -w '%{http_code}' \
    --proxy "$proxy_url" $cacert_flag \
    "https://api.openai.com/v1/models" 2>/dev/null) || true

  case "$status_code" in
    200)
      echo "  Smoke test passed: credential injection is working (HTTP 200)."
      ;;
    401|403)
      echo "  Warning: Gateway is reachable but credential injection may have failed (HTTP $status_code)." >&2
      echo "  Check that OPENAI_API_KEY is valid." >&2
      ;;
    000)
      echo "  Warning: Gateway is unreachable (HTTP 000)." >&2
      echo "  Check that OneCLI is running: docker compose -f $COMPOSE_FILE -f $ONECLI_COMPOSE_FILE ps" >&2
      ;;
    *)
      echo "  Warning: Unexpected HTTP status: $status_code" >&2
      ;;
  esac
}

print_config() {
  # Note: The proxy URL contains your agent access token. Avoid committing it to version control.
  local proxy_url="http://x:${AGENT_ACCESS_TOKEN}@localhost:${ONECLI_GATEWAY_PORT}"

  local site_path="${SITE_DIR:-${ENV_DIR}/onecli-site}"

  cat <<EOF

============================================================
  OneCLI is running. Use these settings in your agent:
============================================================

# Export the proxy and SSL workaround for your agent process:
export HTTPS_PROXY="${proxy_url}"
export PYTHONPATH="${site_path}\${PYTHONPATH:+:\$PYTHONPATH}"

# Run SimLab tasks through the proxy:
HTTPS_PROXY="${proxy_url}" PYTHONPATH="${site_path}" \\
  simlab tasks run \\
  --env ${ENV_NAME} \\
  --task <task_id> \\
  --agent-model <model> \\
  --agent-provider <provider> \\
  --agent-api-key "FAKE_KEY"

# Create a rate-limit rule via the dashboard API:
curl -X POST http://localhost:${ONECLI_DASHBOARD_PORT}/api/rules \\
  -H "Content-Type: application/json" \\
  -d '{"name":"LLM rate limit","hostPattern":"api.openai.com","action":"rate_limit","enabled":true,"rateLimit":5,"rateLimitWindow":"minute"}'

# Dashboard: http://localhost:${ONECLI_DASHBOARD_PORT}
# Gateway:   http://localhost:${ONECLI_GATEWAY_PORT}
============================================================
EOF
}

# ---------------------------------------------------------------------------
# Step 5: Main execution flow
# ---------------------------------------------------------------------------
main() {
  echo "Setting up OneCLI for SimLab environment: ${ENV_NAME}"
  echo ""

  create_onecli_compose
  dc up -d
  wait_for_onecli
  configure_credentials
  smoke_test
  print_config
}

main

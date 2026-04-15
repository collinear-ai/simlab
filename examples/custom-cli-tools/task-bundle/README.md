# Task Bundle

This bundle contains the example task that runs against the custom CLI tools
environment.

- `tasks/workspace_asset_report.json`
  - the task definition
- `verifiers/workspace_asset_report.py`
  - the verifier for the task
- `tasks/example_task.json`
  - the task definition
- `verifiers/custom_coding.py`
  - the verifier for the task

Run it from `cli/simlab` with:

```bash
simlab --environments-dir ./examples env init custom-cli-tools --force --non-interactive

simlab --environments-dir ./examples env up custom-cli-tools

simlab --environments-dir ./examples tasks run \
  --env custom-cli-tools \
  --tasks-dir ./examples/custom-cli-tools/task-bundle \
  --task workspace_asset_report \
  --agent-model gpt-5.2
```

# Custom CLI Tools

This is a small worked example of a coding environment with:

- one startup script that installs extra CLIs
- two mounted fixture files under `/workspace/fixtures`
- one reusable coding skill
- one local task and verifier

For a new environment, start with `simlab env init`. This directory is a
reference example showing what a completed setup looks like after customization.

## Layout

- `env.yaml`
  - the environment config
- `coding/setup/install-tools.sh`
  - installs `pdftotext` and `xlsx2csv`
- `coding/fixtures/`
  - the PDF and spreadsheet mounted into the workspace
- `coding/skills/pdf-xlsx-reporting/SKILL.md`
  - reusable instructions for extracting data from PDF and XLSX files
- `task-bundle/`
  - the local task and verifier

## Run The Example

Run these commands from the `cli/simlab` directory:

```bash
simlab --environments-dir ./examples env init custom-cli-tools --force --non-interactive

simlab --environments-dir ./examples env up custom-cli-tools

simlab --environments-dir ./examples tasks run \
  --env custom-cli-tools \
  --tasks-dir ./examples/custom-cli-tools/task-bundle \
  --task workspace_asset_report \
  --agent-model gpt-5.2
```

When you are done:

```bash
simlab --environments-dir ./examples env down custom-cli-tools
```

## Adapt This Example

To create your own version:

1. Copy this directory or start from `simlab env init`.
2. Replace the install script with the tools you need.
3. Replace the fixture files with your own workspace inputs.
4. Add or edit skills under `coding/skills/`.
5. Replace the task and verifier in `task-bundle/`.

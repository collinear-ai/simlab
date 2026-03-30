# Env-Local Custom Tools

SimLab environments can define additional tool servers under their own
`custom-tools/` directory.

This is useful when you want to:

- add a tool that is not part of the built-in catalog
- keep environment-specific tool definitions close to the env that uses them
- scaffold Harbor-generated tools later without mutating the shared catalog

## Layout

An environment with custom tools looks like this:

```text
environments/my-env/
  env.yaml
  docker-compose.yml
  .env
  custom-tools/
    harbor-main.yaml
```

Custom tool definitions use the same YAML schema as built-in catalog tools.

## Add a Custom Tool

Start from an existing environment:

```bash
simlab env init my-env --template hr
```

Then scaffold a custom tool:

```bash
simlab env custom-tools add my-env harbor-main
```

That command will:

- create `custom-tools/` if needed
- write `custom-tools/harbor-main.yaml`
- add `harbor-main` to `env.yaml`
- regenerate `docker-compose.yml` and other generated env artifacts

If the scaffold already exists, rerun with `--force` to overwrite it.

## Edit and Regenerate

After editing `env.yaml`, `custom-tools/*.yaml`, or `mcp-servers.json`, the
generated files may become stale.

When you later run `simlab env up`, `simlab tasks run`, or `simlab tasks seed`,
SimLab will:

- detect that the generated artifacts are stale
- prompt to regenerate them in interactive sessions
- fail with a clear message in non-interactive flows

To refresh generated files explicitly, rerun:

```bash
simlab env init my-env --force
```

## Inspect Custom Tools

Use `tools info` with `--env` to resolve env-local tools:

```bash
simlab tools info harbor-main --env my-env
```

## Name Rules

- Custom tool names must not shadow built-in tool names.
- Env-local custom tool names must be unique within the environment.
- MCP server names must not conflict with either built-in or env-local tool names.

## Build Support

Service definitions can use either:

- `build: ./path`
- `build:` with `context` and optional `dockerfile`

For Daytona-backed execution, build contexts must stay inside the environment
bundle.

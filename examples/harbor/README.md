# Harbor examples

This directory contains small Harbor tasks that can be run directly with
`simlab tasks run --harbor ...`.

## Included examples

| Path | Purpose |
|------|---------|
| [hello-world/](hello-world/) | Minimal Harbor task that writes `hello.txt` and verifies it with `tests/test.sh`. |

From the `cli/simlab` directory:

```bash
uv run --project cli/simlab simlab tasks run \
  --harbor ./examples/harbor/hello-world \
  --agent-model gpt-5.2 \
  --agent-api-key "$OPENAI_API_KEY"
```

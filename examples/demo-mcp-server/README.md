# Demo MCP server (no auth)

Minimal FastMCP server for local SimLab testing. Exposes Streamable HTTP at `http://0.0.0.0:8081/mcp` with tools `echo`, `ping`, and `get_env`.

Run with Docker:

```bash
docker build -t simlab-demo-mcp .
docker run --rm -p 8081:8081 simlab-demo-mcp
```

Run locally:

```bash
pip install fastmcp
python server.py
```

Use `http://localhost:8081/mcp` in `mcp-servers.json` or copy [mcp-servers-local-demo.json](../mcp-servers-local-demo.json). The full MCP workflow is documented in [QUICKSTART](../../QUICKSTART.md).

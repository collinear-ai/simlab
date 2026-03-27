FROM python:3.12-slim
# Install uv/uvx so command-based MCP servers like `uvx ... mcp-weather` can run
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Install Node.js, npm, and git (git required for uvx --from git+https://...)
RUN apt-get update && apt-get install -y --no-install-recommends git nodejs npm \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY run_gateway.py .
ENV CONFIG_PATH=/config/mcp-gateway-config.json
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=5s --retries=6 --start-period=10s \
  CMD python3 -c "import http.client, sys; conn = http.client.HTTPConnection('localhost', 8080, timeout=5); conn.request('GET', '/mcp'); resp = conn.getresponse(); sys.exit(0 if resp.status < 500 else 1)"
CMD ["python", "-u", "run_gateway.py"]

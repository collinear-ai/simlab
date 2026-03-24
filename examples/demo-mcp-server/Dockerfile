FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8081
EXPOSE 8081
CMD ["python", "-u", "server.py"]

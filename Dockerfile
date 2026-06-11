
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps for native wheels (PyMuPDF wheel ships pre-built; chromadb
# pulls in pyarrow which sometimes needs gcc on slim images).
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install dependencies into an isolated venv we can copy wholesale.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2 — runtime
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Default Ollama target. Overridden at run time when Ollama runs in
    # another container (docker-compose `with-ollama` profile sets this
    # to http://ollama:11434).
    OLLAMA_HOST="http://host.docker.internal:11434" \
    # Hard-cap context window. See app/config.py for why this matters on
    # memory-constrained hosts.
    NUM_CTX="8192"

# curl is used by the HEALTHCHECK below; tini is a tiny init that
# forwards signals so SIGTERM gracefully stops uvicorn.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tini \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user — never run a web service as root.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 app

WORKDIR /app


COPY --from=builder /opt/venv /opt/venv


COPY --chown=app:app app/        ./app/
COPY --chown=app:app eval/       ./eval/
COPY --chown=app:app tests/      ./tests/
COPY --chown=app:app ui/         ./ui/
COPY --chown=app:app run.sh README.md pyproject.toml ./


RUN mkdir -p /app/index /app/Data \
 && chown -R app:app /app/index /app/Data

USER app

EXPOSE 8000


HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health \
        | python -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='ok' else 1)"

# tini → uvicorn so Ctrl+C / docker stop kills the worker cleanly.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/server

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/data/documents /app/data/checkpoints

COPY server/pyproject.toml ./
RUN pip install --no-cache-dir "PyMySQL==1.2.0" ".[dev]"

COPY server/app ./app
COPY server/alembic ./alembic
COPY server/alembic.ini ./alembic.ini
COPY server/scripts ./scripts
COPY sample-data /workspace/sample-data

RUN chown -R appuser:appuser /workspace /app/data
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=5s --start-period=40s --retries=5 CMD curl -fsS http://localhost:8080/api/v1/readiness || exit 1

CMD ["sh", "-c", "umask 077 && chmod 0700 /app/data/checkpoints && exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1"]

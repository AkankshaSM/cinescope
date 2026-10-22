
FROM python:3.12-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-prod.txt .

RUN pip install --no-cache-dir \
    -r requirements-prod.txt


FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

RUN useradd --create-home appuser

COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser static/ ./static/

COPY --chown=appuser:appuser \
    data/movies.db ./data/movies.db

COPY --chown=appuser:appuser \
    chroma_index/ ./chroma_index/

USER appuser

EXPOSE 8080

CMD ["sh", "-c", \
     "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

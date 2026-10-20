FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
# The embedding model goes into the image: the container runs with a read-only filesystem and no download at start.
RUN FASTEMBED_CACHE_PATH=/models /app/.venv/bin/python -c \
    "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
RUN chmod -R a+rX /models

FROM python:3.13-slim
RUN useradd --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=build /app/.venv .venv
COPY --from=build /models /models
COPY *.py schema.sql ./
COPY static static
ENV PATH=/app/.venv/bin:$PATH FASTEMBED_CACHE_PATH=/models HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
USER 10001
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

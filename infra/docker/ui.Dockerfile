# syntax=docker/dockerfile:1.7
# Streamlit UI image (HTTP API only; no system packages required).
FROM python:3.11.13-slim-bookworm

ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG UV_DEFAULT_INDEX=https://pypi.org/simple
ARG UV_VERSION=0.9.7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX} \
    PATH="/app/.venv/bin:${PATH}"

RUN pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" "uv==${UV_VERSION}"

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra ui || uv sync --no-dev --no-install-project --extra ui

COPY src ./src
ENV PYTHONPATH=/app/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ui || uv sync --no-dev --extra ui

EXPOSE 8501
CMD ["streamlit", "run", "src/embodied_ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

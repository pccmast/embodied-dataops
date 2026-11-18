# syntax=docker/dockerfile:1.7
# API / dispatcher image: Python 3.11 with uv-locked dependencies.
#
# IMPORTANT (ADR-0002 / ADR-0009): this image intentionally does **not** run
# `apt-get`; the Debian mirrors are unreachable from the current build network,
# and the API/dispatcher do not need system FFmpeg.  Worker images obtain FFmpeg
# from the pip `imageio-ffmpeg` wheel instead of the distro package.
#
# `datasets.service` builds the LeRobot export synchronously inside the API process
# (SPEC 2.4/2.5), so this image needs the official Writer and the video decoder --
# not just the base lock.  `EXTRAS` therefore defaults to the LeRobot extra; pass
# `--build-arg EXTRAS=""` for an API-only image that never exports.
FROM python:3.11.13-slim-bookworm AS base

ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG UV_DEFAULT_INDEX=https://pypi.org/simple
ARG UV_VERSION=0.9.7
#: Space-separated `uv sync` extras (an exact sync removes anything not selected).
ARG EXTRAS="--extra lerobot"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX} \
    PATH="/app/.venv/bin:${PATH}"

RUN pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" "uv==${UV_VERSION}"

# `EXTRAS` 默认带 `--extra lerobot`，它会引入 evdev；evdev 只有 sdist，需要 gcc 与
# <linux/input.h>（linux-libc-dev），slim 基础镜像两者都没有，否则构建报
# "Failed to build evdev==2.0.0"。仅在确有 extras 时安装，避免给最小镜像加负担。
RUN if [ -n "${EXTRAS}" ]; then \
        apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev linux-libc-dev \
        && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

# Dependency layer (cached while pyproject/uv.lock are unchanged).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${EXTRAS} \
    || uv sync --no-dev --no-install-project ${EXTRAS}

COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./
COPY scripts ./scripts
# Profiles and the VLM prompt are read from the repository at runtime
# (domain/profiles.py, ai/service.py); without this COPY the container starts and
# then fails on the first normalize/inference with "profile file not found".
COPY contracts ./contracts

ENV PYTHONPATH=/app/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev ${EXTRAS} || uv sync --no-dev ${EXTRAS}

# Thumbnails/previews and the aligned v2 export decode video through this wheel;
# installed after the last exact sync so it cannot be pruned.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url "${UV_DEFAULT_INDEX}" "imageio-ffmpeg>=0.6" "av>=15.0"

EXPOSE 8000
CMD ["uvicorn", "embodied_dataops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

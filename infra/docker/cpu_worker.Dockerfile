# syntax=docker/dockerfile:1.7
# CPU worker image: Python 3.11, uv-locked deps, optional official LeRobot API /
# YOLO11n.  No `apt-get` by default: FFmpeg comes from the pip `imageio-ffmpeg`
# wheel (PyAV is also installed), which is the documented provider fallback in
# ADR-0002 / ADR-0009.  Set INSTALL_SYSTEM_FFMPEG=1 on networks that can reach
# the distro apt mirrors.
#
# Dependency order matters (`uv sync` is an *exact* sync: everything not selected is
# removed from the environment):
#
#   1. layer 1: the base lock, no project
#   2. layer 2: the project + every requested extra, in **one** sync
#   3. media/vision packages that are not lock-file dependencies, installed last
#
# A final `uv sync` without the extras used to delete LeRobot/Ultralytics (and the
# FFmpeg wheel) from an image that had just installed them.
FROM python:3.11.13-slim-bookworm

ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG UV_DEFAULT_INDEX=https://pypi.org/simple
ARG UV_VERSION=0.9.7
ARG INSTALL_SYSTEM_FFMPEG=0
#: Opt-in extra stacks (compose maps INSTALL_LEROBOT / INSTALL_VISION onto these).
ARG INSTALL_LEROBOT=0
ARG INSTALL_VISION=0
#: Space-separated `uv sync` extras, e.g. "--extra lerobot --extra vision".
ARG EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX} \
    PATH="/app/.venv/bin:${PATH}"

RUN pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" "uv==${UV_VERSION}" \
    && if [ "$INSTALL_SYSTEM_FFMPEG" = "1" ]; then \
         apt-get update && apt-get install -y --no-install-recommends ffmpeg \
         && rm -rf /var/lib/apt/lists/*; \
       fi

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./

# Base dependency layer first so the common set is cached.
# uv cache is mounted as a BuildKit cache (not part of the image): without it the
# downloaded wheels/unpacked archives end up in the layer and roughly double the
# image size (measured: 13 GB cache on top of a 8.7 GB venv for the CUDA build).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project || uv sync --no-dev --no-install-project

COPY src ./src
COPY contracts ./contracts
COPY scripts ./scripts
COPY alembic ./alembic
COPY alembic.ini ./
ENV PYTHONPATH=/app/src

# One sync installs the project *and* the requested extras; nothing is pruned later.
# INSTALL_LEROBOT / INSTALL_VISION are the compose-facing switches; EXTRAS is the
# escape hatch for an arbitrary combination.
RUN --mount=type=cache,target=/root/.cache/uv set -eu; \
    SELECTED="${EXTRAS}"; \
    if [ "$INSTALL_LEROBOT" = "1" ]; then SELECTED="$SELECTED --extra lerobot"; fi; \
    if [ "$INSTALL_VISION" = "1" ]; then SELECTED="$SELECTED --extra vision"; fi; \
    echo "uv sync extras:${SELECTED}"; \
    if [ -n "$SELECTED" ]; then \
        echo "installing C build deps (evdev compiles from sdist, needs gcc + linux/input.h)"; \
        apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev linux-libc-dev \
        && rm -rf /var/lib/apt/lists/*; \
    fi; \
    uv sync --frozen --no-dev ${SELECTED} || uv sync --no-dev ${SELECTED}

# FFmpeg replacement binary (imageio-ffmpeg ships a static ffmpeg; ffprobe is
# emulated with PyAV by the processing wrapper).  Installed after the last sync so
# an exact sync cannot remove it.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url "${UV_DEFAULT_INDEX}" "imageio-ffmpeg>=0.6" "av>=15.0"

# torchcodec (a LeRobot dependency) opens FFmpeg by *standard* SONAME --
# ``libavutil.so.59``, ``libavcodec.so.61`` ... -- while the ``av`` wheel ships the very
# same libraries auditwheel-renamed inside ``av.libs/libavutil-<hash>.so.59``.  With
# INSTALL_SYSTEM_FFMPEG=0 there is no distro FFmpeg to satisfy those lookups, so every
# LeRobot video read dies with "Could not load libtorchcodec" (measured: torchcodec
# 0.10.0 + FFmpeg 7 inside this image, `libavutil.so.56: cannot open shared object`).
# Building the SONAME links here keeps apt out of the build path (ADR-0009) and leaves
# the renamed dependencies in place, which the loader still needs for their own deps.
# Fail-closed: a PyAV wheel layout change breaks the build, not the runtime.
RUN /app/.venv/bin/python - <<'PY'
import pathlib
import re

libs = pathlib.Path("/app/.venv/lib/python3.11/site-packages/av.libs")
if not libs.is_dir():
    raise SystemExit(f"av.libs not found at {libs}: the PyAV wheel layout changed")
out = pathlib.Path("/opt/ffmpeg-soname")
out.mkdir(parents=True, exist_ok=True)
made: list[str] = []
for entry in libs.iterdir():
    match = re.match(r"(lib[a-zA-Z0-9_]+)-[0-9a-f]{8}(\.so\.[0-9]+)", entry.name)
    if not match:
        continue
    link = out / (match.group(1) + match.group(2))
    if not link.exists():
        link.symlink_to(entry)
        made.append(link.name)
required = {"libavutil", "libavcodec", "libavformat"}
missing = required - {name.split(".so.")[0] for name in made}
if missing:
    raise SystemExit(f"missing FFmpeg SONAMEs: {sorted(missing)}")
print(f"ffmpeg soname links: {len(made)}")
PY

#: The link directory first, then av.libs: the renamed dependencies (libaom-<hash>.so.3,
#: libSvtAv1Enc-<hash>.so.3, ...) are only resolvable under their original names.
ENV LD_LIBRARY_PATH=/opt/ffmpeg-soname:/app/.venv/lib/python3.11/site-packages/av.libs

# ``ultralytics`` (the ``vision`` extra) depends on ``opencv-python`` -- the *GUI*
# build -- which needs ``libGL.so.1`` and ``libglib-2.0.so.0``.  This image deliberately
# installs no distro GUI libraries (ADR-0009), and both OpenCV wheels own the same
# ``cv2/`` directory, so the GUI build silently replaced the headless one declared in
# ``pyproject.toml``.  Measured symptom inside the container: ``BUILD_PREVIEW`` and
# ``DETECT_OBJECTS`` were consumed by the worker and failed with ``HANDLER_UNAVAILABLE``
# / ``ImportError: libGL.so.1: cannot open shared object file``, which also leaves
# ``GET /episodes/{id}/preview`` at 409.
# Drop the GUI build and restore the headless files, then prove ``cv2`` imports: a
# rebuild that reintroduces the conflict must fail here, not at task time.
RUN --mount=type=cache,target=/root/.cache/uv set -eu; \
    uv pip uninstall opencv-python || true; \
    uv pip install --index-url "${UV_DEFAULT_INDEX}" --no-deps --force-reinstall \
        "opencv-python-headless>=4.10,<5"; \
    /app/.venv/bin/python -c "import cv2, pathlib; print('cv2 ok', cv2.__version__, pathlib.Path(cv2.__file__).parent)"

# Opt-in CUDA torch for the GPU overlay (`docker compose -f docker-compose.yml -f
# docker-compose.gpu.yml build worker-vision`).  PyPI's default wheel is CPU-only, so
# "the host has a GPU" is not enough -- the image must carry a CUDA build.  The wheel
# index is pinned by the build, never resolved at runtime.
ARG TORCH_INDEX_URL=""
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$TORCH_INDEX_URL" ] && [ "$INSTALL_VISION" = "1" ]; then \
        echo "installing CUDA torch from $TORCH_INDEX_URL"; \
        uv pip install --reinstall --index-url "$TORCH_INDEX_URL" torch torchvision; \
    fi

CMD ["python", "-m", "embodied_dataops.workflow.worker", "--pool", "cpu", "--concurrency", "2"]

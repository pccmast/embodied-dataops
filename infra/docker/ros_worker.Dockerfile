# syntax=docker/dockerfile:1.7
# ROS 2 Jazzy worker: only this image may read MCAP with rosbag2_py.
#
# `ros:jazzy-ros-base` already ships `rosbag2_py` and the MCAP storage plugin
# (verified on 2026-10-29: F05 topics read natively: 515 action / 1020 joint /
# 300 image messages), so this image needs **no apt install** of ROS packages.
# Application Python dependencies are installed into the distribution's Python
# 3.12 from the exported uv lock, so rosbag2_py and the app share one
# interpreter.  The main project package is not pip-installed; it is imported via
# PYTHONPATH (ADR-0003 / ADR-0009).
FROM ros:jazzy-ros-base

ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG UV_DEFAULT_INDEX=https://pypi.org/simple
ARG UV_VERSION=0.9.7

# Debian/Ubuntu 的 /usr 是 externally managed（PEP 668），uv pip --system 必须显式放行。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX} \
    UV_BREAK_SYSTEM_PACKAGES=1

# `ros:jazzy-ros-base` 不带 pip（实测 `python3 -m pip` → "No module named pip"），
# 只补 pip 本身；ROS 侧依赖仍由镜像自带，不额外 apt 安装 ROS 包。
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url "${PYPI_INDEX_URL}" "uv==${UV_VERSION}"

WORKDIR /app
# ``requirements-v2.txt`` is an `uv export --no-emit-project` of the app's runtime
# dependency set: it deliberately does NOT contain `-e .`, because at this point in
# the build the project sources are not copied yet (that ordering made the image
# fail with "pyproject.toml not found").  The application is imported via PYTHONPATH.
COPY services/ros2_adapter/requirements-v2.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --index-url "${UV_DEFAULT_INDEX}" -r requirements-v2.txt

COPY src ./src
# Profiles and the VLM prompt are loaded from the repository at runtime.
COPY contracts ./contracts
COPY services/ros2_adapter ./services/ros2_adapter
ENV PYTHONPATH=/app/src:/app/services/ros2_adapter

# No published ports in docker-compose.yml: the ROS worker is internal-only.
CMD ["python3", "-m", "embodied_dataops.workflow.worker", "--pool", "ros2", "--concurrency", "1"]

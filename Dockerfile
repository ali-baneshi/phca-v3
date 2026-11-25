# =============================================================================
# PHCA — Docker image
# =============================================================================
#   docker build --target phca-core   -t phca:core   .   numpy + dev tools
#   docker build --target phca-mujoco -t phca:mujoco .   + MuJoCo environments
#
# Dependencies come from pyproject.toml extras; there is no separate pin list.
# =============================================================================

FROM python:3.11-slim AS phca-core

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[dev,stats]"

ENV MUJOCO_GL=disabled
ENTRYPOINT ["python"]

# ── MuJoCo runtime ───────────────────────────────────────────────────────────

FROM phca-core AS phca-mujoco

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglfw3 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -e ".[mujoco]"

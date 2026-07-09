# =============================================================================
# PHCA v3.0 — Multi-stage Docker image
# =============================================================================
# Build with:
#   docker build --target phca-core   -t phca:core   .   (default, ~400 MB)
#   docker build --target phca-mujoco -t phca:mujoco .   (~550 MB)
#   docker build --target phca-full   -t phca:full   .   (~750 MB)
# =============================================================================

FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements*.txt ./
# Install core deps WITHOUT pgmpy (which pulls torch + CUDA ~2 GB).
RUN sed '/^pgmpy/d' requirements.txt > /tmp/req_no_pgmpy.txt \
    && pip install --no-cache-dir -r /tmp/req_no_pgmpy.txt -r requirements-dev.txt

# pgmpy 1.x pulls in torch + CUDA (~2 GB) via optional deps.
# PHCA only uses DiscreteBayesianNetwork/TabularCPD/VariableElimination
# which run fine without torch/pyro-ppl.  Install pgmpy without deps and
# manually supply the few non-CUDA transitive dependencies that aren't
# already in requirements.txt.
RUN pip install --no-cache-dir pgmpy==1.0.0 --no-deps \
    && pip install --no-cache-dir "opt_einsum>=3.3" "tqdm>=4.64" "pandas>=2.0" "statsmodels>=0.14" "pyro-api>=0.1.1" "scikit-learn>=1.4"

COPY . .

# ── Core runtime (CI, stress tests, pytest) ──────────────────────────────────

FROM python:3.11-slim AS phca-core

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

WORKDIR /app
ENV PYTHONPATH=/app/python \
    MUJOCO_GL=disabled \
    MPLBACKEND=Agg
ENTRYPOINT ["python"]

# ── MuJoCo runtime (MuJoCo gate, nightly with MuJoCo envs) ──────────────────

FROM phca-core AS phca-mujoco

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglfw3 \
    libosmesa6 \
    libglew-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements-mujoco.txt

# ── Full runtime (MuJoCo + PyQt5 Observatory) ────────────────────────────────

FROM phca-mujoco AS phca-full

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libfontconfig1 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libx11-xcb1 \
    libxrender1 \
    libegl1 \
    libegl1-mesa \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

#!/usr/bin/env bash
# PHCA overnight / multi-hour diagnosis harness (G2-INV-05 + L4 + Φ-IQ).
#
# Does NOT flip disable_blended_scorer default. Collects JSON under logs/overnight_*.
# Safe to leave running for several hours (typical: ~4–10h on a desktop CPU).
#
# Usage:
#   chmod +x scripts/overnight_diagnosis_harness.sh
#   mkdir -p logs
#   nohup env MUJOCO_GL=disabled PYTHONPATH=python \
#     ./scripts/overnight_diagnosis_harness.sh \
#     > logs/overnight_harness_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo $! > logs/overnight_harness.pid
#   tail -f logs/overnight_harness_*.log   # optional monitor
#
# Faster (~1–2h) smoke-power variant:
#   SEEDS=10 CYCLES=100 ./scripts/overnight_diagnosis_harness.sh
#
# Full statistical power (project standard):
#   SEEDS=30 CYCLES=200 ./scripts/overnight_diagnosis_harness.sh
#
# Note: benchmark.py expects comma-separated levels (e.g. 0,1,2), not ranges (0-2).
set -u
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MUJOCO_GL="${MUJOCO_GL:-disabled}"
export PYTHONPATH="${PYTHONPATH:-}:python"

SEEDS="${SEEDS:-30}"
CYCLES="${CYCLES:-200}"
BASE_SEED="${BASE_SEED:-42}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-logs/overnight_${STAMP}}"
mkdir -p "$OUT_DIR"

run() {
  local name="$1"
  shift
  local out="$OUT_DIR/${name}.json"
  local log="$OUT_DIR/${name}.log"
  echo ""
  echo "============================================================"
  echo "[$(date -Iseconds)] START $name"
  echo "  cmd: $*"
  echo "  out: $out"
  echo "============================================================"
  local t0
  t0=$(date +%s)
  set +e
  "$@" --output "$out" >"$log" 2>&1
  local rc=$?
  set -e
  local t1
  t1=$(date +%s)
  echo "[$(date -Iseconds)] END $name rc=$rc elapsed=$((t1 - t0))s" | tee -a "$OUT_DIR/manifest.txt"
  if [[ $rc -ne 0 ]]; then
    echo "  WARN: $name failed (rc=$rc); continuing. See $log" | tee -a "$OUT_DIR/manifest.txt"
  fi
  return 0
}

{
  echo "overnight_diagnosis_harness"
  echo "stamp=$STAMP"
  echo "seeds=$SEEDS cycles=$CYCLES base_seed=$BASE_SEED"
  echo "host=$(hostname) nproc=$(nproc 2>/dev/null || echo '?')"
  echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "started=$(date -Iseconds)"
} | tee "$OUT_DIR/manifest.txt"

# ---------------------------------------------------------------------------
# 1) Causal geometry default — project gate levels (primary residual: G2-INV-05)
# ---------------------------------------------------------------------------
run causal_5x5_l2l3_geometry \
  python scripts/phca_causal_eval.py \
    --levels level2,level3 --cycles "$CYCLES" --seeds "$SEEDS" \
    --base-seed "$BASE_SEED" --grid-size 5 --use-mlp --gate

run causal_10x10_l2l3_geometry \
  python scripts/phca_causal_eval.py \
    --levels level2,level3 --cycles "$CYCLES" --seeds "$SEEDS" \
    --base-seed "$BASE_SEED" --grid-size 10 --use-mlp --gate

# ---------------------------------------------------------------------------
# 2) Blended opt-in contrast (same seeds) — H3 evidence at power; still opt-in
# ---------------------------------------------------------------------------
run causal_5x5_l2_blended \
  python scripts/phca_causal_eval.py \
    --levels level2 --cycles "$CYCLES" --seeds "$SEEDS" \
    --base-seed "$BASE_SEED" --grid-size 5 --use-mlp --gate \
    --enable-blended-scorer

run causal_10x10_l2_blended \
  python scripts/phca_causal_eval.py \
    --levels level2 --cycles "$CYCLES" --seeds "$SEEDS" \
    --base-seed "$BASE_SEED" --grid-size 10 --use-mlp --gate \
    --enable-blended-scorer

# ---------------------------------------------------------------------------
# 3) Ablation matrix A/B/C (shared seeds) — planner vs learn-off vs blended
# ---------------------------------------------------------------------------
DIAG_SEEDS="${DIAG_SEEDS:-$SEEDS}"
DIAG_CYCLES="${DIAG_CYCLES:-$CYCLES}"

run diagnosis_5x5_l2_abc \
  python scripts/diagnose_causal_ablation.py \
    --level level2 --cycles "$DIAG_CYCLES" --seeds "$DIAG_SEEDS" \
    --base-seed "$BASE_SEED" --grid-size 5 --use-mlp

run diagnosis_10x10_l2_abc \
  python scripts/diagnose_causal_ablation.py \
    --level level2 --cycles "$DIAG_CYCLES" --seeds "$DIAG_SEEDS" \
    --base-seed "$BASE_SEED" --grid-size 10 --use-mlp

# ---------------------------------------------------------------------------
# 4) L4 forgetting + dual PE (geometry confound residual G4-INV-03)
# ---------------------------------------------------------------------------
run level4_30s \
  python scripts/benchmark_level4.py \
    --tasks 10 --task-cycles 80 --seeds "$SEEDS"

# ---------------------------------------------------------------------------
# 5) Φ-IQ L0–L2 labeled report (planner contamination honesty)
# ---------------------------------------------------------------------------
run phi_iq_l0l2 \
  python scripts/benchmark.py \
    --levels=0,1,2 --cycles="$CYCLES" --use-mlp --seeds="$SEEDS"

# ---------------------------------------------------------------------------
# 6) Optional viewport causal (commented by default — add hours)
# Uncomment if you want the full night:
# run causal_10x10_viewports_geometry \
#   python scripts/phca_causal_eval.py \
#     --levels viewport1,viewport2,viewport3 --cycles "$CYCLES" --seeds "$SEEDS" \
#     --base-seed "$BASE_SEED" --grid-size 10 --use-mlp --gate

echo "finished=$(date -Iseconds)" | tee -a "$OUT_DIR/manifest.txt"
echo "OUT_DIR=$OUT_DIR"
ls -la "$OUT_DIR" | tee -a "$OUT_DIR/manifest.txt"
echo "Done. Analyze JSON under $OUT_DIR (see manifest.txt)."

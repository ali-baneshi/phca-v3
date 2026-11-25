#!/usr/bin/env bash
# Offline gate for the minimization work.
#
# Runs on the interpreter that has the project's runtime + test dependencies.
# This repository is pinned to >=3.11,<3.13 but the only complete local
# test environment is the system python (3.14), so the wheel build ignores
# the requires-python marker. The final install check happens in Phase 6.
#
# Usage: scripts/gate.sh [--fast]
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
export PYTHONPATH="python${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=disabled

echo "[gate] import phca"
"$PY" -c "import phca; print('  import ok')"

if [ "${1:-}" != "--fast" ]; then
    echo "[gate] full test suite"
    "$PY" -m pytest python/tests/ python/phca/ -q -p no:cacheprovider \
        --timeout=60 --benchmark-skip \
        --ignore=python/tests/test_mujoco_env.py \
        --ignore=python/tests/test_cycle_with_mujoco.py \
        --ignore=python/tests/test_continuous_actions.py
fi

echo "[gate] wheel build"
# Build from a clean export of the working tree (tracked + new, unignored files)
# so stale local build/ or *.egg-info directories cannot leak into the wheel.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/src"
git ls-files -z --cached --others --exclude-standard \
    | while IFS= read -r -d '' f; do [ -e "$f" ] && printf '%s\0' "$f"; done \
    | tar --null -T - -cf - | tar -xf - -C "$WORK/src"
"$PY" -m pip wheel "$WORK/src" --no-deps --no-build-isolation --ignore-requires-python \
    -w "$WORK/dist" -q 2>/dev/null

WHL="$(find "$WORK/dist" -name '*.whl' -print -quit)"
if [ -z "$WHL" ]; then
    echo "[gate] FAIL: no wheel produced" >&2
    exit 1
fi
echo "  wheel ok: $(du -h "$WHL" | cut -f1) $(basename "$WHL")"

echo "[gate] wheel contents"
"$PY" - "$WHL" <<'PY'
import sys
import zipfile

names = zipfile.ZipFile(sys.argv[1]).namelist()
leaks = [n for n in names if "/tests/" in n or n.startswith(("attic/", "logs/", "docs/"))
         or "/monitoring/" in n]
if leaks:
    print(f"  FAIL: {len(leaks)} unwanted entries, e.g. {leaks[:3]}", file=sys.stderr)
    sys.exit(1)
print(f"  ok: {len(names)} entries, no tests/attic/logs/docs/monitoring")
PY

echo "[gate] PASS"

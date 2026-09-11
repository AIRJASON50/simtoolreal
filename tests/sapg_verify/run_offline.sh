#!/usr/bin/env bash
# One-command runner for the GPU-level SAPG OFFLINE verification suite.
# Static + 2/3-proc CPU gloo. No GPU, no IsaacSim. Run on the cluster dev pod:
#   bash tests/sapg_verify/run_offline.sh
# Override the interpreter with PYBIN=... (default = the SimToolReal isaacsim venv).
set -euo pipefail

# repo root = dir containing rl_games/rl_games/common/a2c_common.py (marker-walk)
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$here"
while [[ "$root" != "/" && ! -f "$root/rl_games/rl_games/common/a2c_common.py" ]]; do
  root="$(dirname "$root")"
done
if [[ ! -f "$root/rl_games/rl_games/common/a2c_common.py" ]]; then
  echo "ERROR: could not locate worktree root from $here" >&2
  exit 2
fi

PYBIN="${PYBIN:-/workspace/simtoolreal/.venv_isaacsim/bin/python}"
if [[ ! -x "$PYBIN" ]]; then
  echo "WARN: $PYBIN not executable; falling back to 'python' on PATH" >&2
  PYBIN="python"
fi

cd "$root"
echo "== repo root: $root"
echo "== interpreter: $PYBIN"
echo "== $($PYBIN -c 'import torch; print("torch", torch.__version__)' 2>&1 | head -1)"
echo

# Run the offline suite + the shipped plumbing tests together.
exec "$PYBIN" -m pytest \
  tests/sapg_verify/test_offline_suite.py \
  tests/test_gpu_level_sapg.py \
  -v -ra "$@"

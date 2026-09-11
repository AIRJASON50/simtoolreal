#!/usr/bin/env bash
# =============================================================================
# SC5c: GPU-level SAPG throughput + peak-memory no-regression vs naive DDP.
# Live 2-card A/B on the cluster (needs >=2 same-node GPUs; gloo tcp://127.0.0.1).
#
# Runs BOTH arms N_REPEAT times, INTERLEAVED on the SAME pinned 2 GPUs, so
# thermal / neighbour noise hits both arms symmetrically and the verdict script
# can quantify within-arm variance vs between-arm gap. The ONLY differing knob
# is gpu_level_sapg (+ its forced block size); every other config is byte-matched.
#
# PREREQ (mandatory to close the no-op / broken-impl false-pass holes):
#   git apply tests/sc5c/sc5c_instrumentation.patch      # from repo root
#   ... and the CPU-gloo correctness gate must be GREEN on this SHA:
#   .venv_isaacsim/bin/python -m pytest tests/test_gpu_level_sapg.py \
#       tests/test_gpu_level_sapg_augment_equiv.py -q
#   (the verdict script re-runs this gate and REFUSES to emit PASS if it fails.)
#
# SCOPE HONESTY (do not remove this banner):
#   * SimToolRealSAPG.yaml uses rnn: lstm. _gather_rollout asserts `not self.is_rnn`,
#     so gpu-level SAPG CANNOT run the shipped LSTM backbone. This harness runs a
#     MLP-only (rnn-disabled) SAPG config so BOTH arms exercise the same non-RNN
#     path. => The measured ratio scopes the NON-RNN SAPG config, NOT the production
#     LSTM backbone. gpu-level SAPG is currently UNUSABLE with the real LSTM config;
#     that is a separate known blocker, not something SC5c can paper over.
#   * central_value_config is asymmetric and its buffer is NOT gathered by
#     _gather_rollout (only sharded downstream). To keep the "one all-gather"
#     overhead model faithful and both arms apples-to-apples, we DISABLE
#     central_value in BOTH arms (STR_DISABLE_CV=1, default on). Set 0 to keep it,
#     but then the overhead model is not "one all-gather" and the ratio is looser.
#   * This bounds ws=2 ONLY. all-gather cost and the ws*local transient grow with
#     world_size; a ws=2 PASS is NOT evidence for the 8/16-card regime. Set WS
#     higher (quota permitting) to add a point; the verdict tags the ws it saw.
# =============================================================================
set -euo pipefail

# ---- repo / interpreter -----------------------------------------------------
REPO=${REPO:-/workspace/simtoolreal}
cd "$REPO"
PY=${PY:-.venv_isaacsim/bin/python}
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TASK=${TASK:-Isaacsimenvs-SimToolReal-Direct-v0}
AGENT=${AGENT:-rl_games_sapg_cfg_entry_point}

# ---- matched config (the ONLY differing knob is gpu_level_sapg) --------------
WS=${WS:-2}                       # world size = #GPUs = #global SAPG blocks (gpu-level arm)
NENV=${NENV:-12288}              # PER-CARD envs (matched local rollout workload)
HORIZON=${HORIZON:-16}
SEED=${SEED:-42}
MINIBATCH=${MINIBATCH:-98304}
MINI_EPOCHS=${MINI_EPOCHS:-2}
WARMUP=${WARMUP:-40}
MEASURE_EPOCHS=${MEASURE_EPOCHS:-80}
EPOCHS=$(( WARMUP + MEASURE_EPOCHS ))
N_REPEAT=${N_REPEAT:-3}          # >=3 so verdict can estimate within-arm variance
# naive-DDP intra-card block size: NENV must be a multiple. 2048 -> 6 blocks/card.
NAIVE_BLOCK=${NAIVE_BLOCK:-2048}
STR_DISABLE_CV=${STR_DISABLE_CV:-1}   # 1 = drop central_value in BOTH arms (apples-to-apples)
OUT=${OUT:-$REPO/sc5c_out}
mkdir -p "$OUT"

# ---- GPU pinning: use the SAME two GPUs for every run -----------------------
GPUS=${GPUS:-0,1}
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if [ "${#GPU_ARR[@]}" -lt "$WS" ]; then
  echo "FATAL: need $WS GPUs, got ${#GPU_ARR[@]} in GPUS=$GPUS" >&2; exit 2
fi

# rnn-disable + optional cv-disable + non-adaptive knobs shared by both arms.
# We keep lr_schedule=adaptive OFF-critical? No: adaptive LR broadcasts from rank0
# identically in both arms, so it does not bias the ratio. We DO force the MLP
# path by nulling the rnn block (Hydra: set rnn to null via ~ delete).
COMMON_OVR=(
  "env.scene.num_envs=$NENV"
  "agent.params.seed=$SEED"
  "agent.params.config.multi_gpu=True"
  "agent.params.config.horizon_length=$HORIZON"
  "agent.params.config.seq_length=$HORIZON"
  "agent.params.config.minibatch_size=$MINIBATCH"
  "agent.params.config.mini_epochs=$MINI_EPOCHS"
  "agent.params.config.max_epochs=$EPOCHS"
  "~agent.params.network.rnn"                       # DELETE rnn block -> MLP-only (non-RNN scope)
)
if [ "$STR_DISABLE_CV" = "1" ]; then
  COMMON_OVR+=( "~agent.params.config.central_value_config" )  # symmetric: both arms drop CV
fi

# -----------------------------------------------------------------------------
# launch(): one arm, one repeat. Spawns WS per-rank procs on the pinned GPUs,
# each writing to its own hydra run dir under $rundir/rank<r>. We capture the
# rank-0 hydra dir (holds the summaries) into a manifest line for the verdict.
# -----------------------------------------------------------------------------
launch () {
  local arm="$1"; local rep="$2"; shift 2
  local extra=("$@")
  local tag="${arm}_rep${rep}"
  echo "=== [SC5c] launching arm=$arm rep=$rep (WS=$WS NENV=$NENV H=$HORIZON) ==="
  local pids=()
  for r in $(seq 0 $((WS-1))); do
    local dev="${GPU_ARR[$r]}"
    local rundir="$OUT/$tag/rank$r"
    mkdir -p "$rundir"
    # SC5C_MEASURE=1 turns on the patched instrumentation (peak mem + diagnostics).
    LOCAL_RANK=$r RANK=$r WORLD_SIZE=$WS \
    SC5C_MEASURE=1 SC5C_WARMUP=$WARMUP \
    CUDA_VISIBLE_DEVICES="$dev" \
    $PY isaacsimenvs/train.py \
      --task "$TASK" --agent "$AGENT" --headless \
      --rl_device "cuda:0" --sim_device "cuda:0" \
      hydra.run.dir="$rundir" \
      "${COMMON_OVR[@]}" \
      "agent.params.config.name=${r}_${tag}" \
      "${extra[@]}" \
      > "$rundir/train.log" 2>&1 &
    pids+=($!)
  done
  # (Because CUDA_VISIBLE_DEVICES pins each proc to one physical GPU, rl_device=cuda:0
  #  inside the proc == the pinned device; local_rank=0 device index inside the proc.
  #  torch.cuda.max_memory_allocated(local_rank) then reads the correct visible device.)
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  if [ "$rc" -ne 0 ]; then
    echo "FATAL: arm=$arm rep=$rep had a non-zero rank (rc=$rc). Check $OUT/$tag/rank*/train.log" >&2
    # A common expected failure here is the RNN assert if you forgot ~network.rnn.
    grep -l "does not support RNN" "$OUT/$tag"/rank*/train.log 2>/dev/null && \
      echo "  -> hit the RNN assert: the ~agent.params.network.rnn delete did not apply." >&2
    exit "$rc"
  fi
  # record rank-0 hydra dir (summaries live at <rundir>/<name>/summaries)
  echo "$arm rep=$rep rank0_dir=$OUT/$tag/rank0 expname=${tag}" >> "$OUT/manifest.txt"
}

: > "$OUT/manifest.txt"

# ---- interleaved schedule: A B A B A B  (not AAA BBB) ------------------------
for rep in $(seq 1 "$N_REPEAT"); do
  # Arm A: naive DDP baseline (intra-card multi-block SAPG, grad-avg only)
  launch naiveddp "$rep" \
    "agent.params.config.gpu_level_sapg=False" \
    "agent.params.config.expl_coef_block_size=$NAIVE_BLOCK"

  # Arm B: gpu-level SAPG (one block/card; _setup forces block_size=num_actors,
  # so expl_coef_block_size here is ignored -- pass NENV to satisfy the divisibility
  # assert should the flag ever fail to engage, which the verdict then catches).
  launch gpulevel "$rep" \
    "agent.params.config.gpu_level_sapg=True" \
    "agent.params.config.expl_coef_block_size=$NENV"
done

echo
echo "=== [SC5c] all runs done. Verdict: ==="
$PY tests/sc5c/sc5c_verdict.py \
  --out "$OUT" \
  --ws "$WS" \
  --expected-frames $(( NENV * HORIZON * WS )) \
  --naive-block "$NAIVE_BLOCK" \
  --nenv "$NENV" \
  --minibatch "$MINIBATCH" \
  --warmup "$WARMUP" \
  --repo "$REPO" \
  --py "$PY"

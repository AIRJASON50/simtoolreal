# GPU-level SAPG verification suite — run instructions

This suite proves the GPU-level SAPG implementation is numerically equivalent to a
single process running world_size blocks, that grad averaging is lossless, and that
every piece of plumbing conforms to the SAPG principle. It is split into an OFFLINE
group (this file's primary target: static + 2/3-proc CPU gloo, NO GPU / NO IsaacSim)
and a LIVE group (>=2/>=3 GPU, deferred).

## Environment

No torch exists on the local dev box; run on the cluster dev pod `str-isaac-dev`,
which has the SimToolReal venv with rl_games editable-installed:

    interpreter = /workspace/simtoolreal/.venv_isaacsim/bin/python

The offline suite needs ONLY torch + rl_games + pyyaml (import chain pulls
gym / tensorboardX / omegaconf, NOT IsaacSim). It runs on any host with that venv.

## OFFLINE group (static + 2/3-proc CPU gloo) — the deliverable

One command, from the worktree root:

    cd /workspace/simtoolreal && \
      .venv_isaacsim/bin/python -m pytest tests/sapg_verify/test_offline_suite.py -v

Sub-selecting by claim id (pytest -k):

    # pure static / AST (no dist, sub-second): SC1a SC1b SC1c SC2a SC2b SC2c SC3b SC4b SC5a SC5b SC6a SC6b
    .venv_isaacsim/bin/python -m pytest tests/sapg_verify/test_offline_suite.py \
        -k "SC1 or SC2 or SC3b or SC4b or SC5 or SC6" -v

    # 2/3-proc CPU gloo (spawns gloo procs on 127.0.0.1): SC3a SC4a SC4c
    .venv_isaacsim/bin/python -m pytest tests/sapg_verify/test_offline_suite.py \
        -k "SC3a or SC4a or SC4c" -v

The shipped plumbing tests still pass alongside:

    .venv_isaacsim/bin/python -m pytest tests/test_gpu_level_sapg.py tests/sapg_verify/ -v

Gloo ports: 29650..29654 on 127.0.0.1. If a port is busy, bump `GLOO_PORT_BASE`
in test_offline_suite.py.

## LIVE group (deferred — NOT part of the offline verdict)

These need real GPUs + IsaacSim on the cluster and cannot give bitwise tolerances.
They corroborate what the offline group already proves structurally.

1. SC2b/SC2c live off_policy_contrib > 0 (leader genuinely consumes follower
   experience). Requires LOG_OFF_POLICY_GRADS=1 (else off_policy_contrib is a
   hardcoded 0 by construction — proven statically by SC2c). Also needs an MLP
   config, NOT the shipped SimToolRealSAPG.yaml (LSTM), because gpu_level_sapg
   rejects RNN (proven by SC6b). Launch with volcano on >=2 GPUs:

       LOG_OFF_POLICY_GRADS=1 <train launch with gpu_level_sapg=True, rnn removed>
       # then read tensorboard scalar auxiliary_stats/off_policy_contrib;
       # assert some logged value is in (0, 1], not identically 0.

2. SC5b comm-flat-vs-off_policy_ratio runtime differential. MUST run at
   world_size >= 3, because gpu-level forces num_blocks == world_size and
   num_repeat = min(num_blocks, off_policy_ratio+1); at ws=2 off_policy_ratio has
   ZERO effect (min(2, .) == 2 for ratio in {1,2}) so a 2-card differential is
   vacuous. On >=3 GPUs, vary off_policy_ratio in {1,2} (both < num_blocks) and
   confirm per-epoch collective count/bytes are identical while local materialized
   batch grows. NOTE: the per-minibatch grad all_reduce count scales with the
   augmented dataset size, so the "comm independent of off_policy_ratio" property
   holds for the ROLLOUT-GATHER + AUGMENT-BROADCAST collectives only, not for the
   optimization-phase grad all_reduce — see the SC5b scope note.

3. The 8-card training-gain A/B (naive DDP vs gpu-level phase-transition timing)
   is explicitly out of scope for this suite.

## What the offline verdict does and does NOT prove

Proves (offline, bitwise/structural): per-rank block assignment, block-b==rank-b
global layout, context swap num_blocks==world_size, lf follower->exploit relabel +
off_policy_mask, PPO clipped-IS weighting + dataset routing + env-gated contrib
metric, rank-ordered all-gather + leader-draws-cross-rank, naive-DDP contrast,
bitwise gather==single-process aggregate, disjoint equal contiguous horizon-aligned
shard, lossless SUM/ws averaging, single-GPU byte-identity gate, comm not scaling
with off_policy_ratio (static), local frame accounting, RNN scope rejection.

Does NOT prove (deferred to LIVE): live off_policy_contrib>0, real NCCL wire bytes,
runtime comm-vs-ratio at ws>=3, training benefit, IsaacSim rollout fidelity, the
production LSTM config (gpu_level_sapg rejects RNN — an MLP config is required).

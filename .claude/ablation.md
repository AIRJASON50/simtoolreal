# Ablation: GPU-level SAPG for SimToolReal (flat, gather+reuse-augment+shard)

SSOT for this worktree. Base = origin/main 313d5aea. Branch jason/gpu-level-sapg.
Requirement (user): implement gpu-level SAPG in BOTH SimToolReal AND GMT (identical rl_games fork).

## Direction
Replace naive-DDP SAPG (each GPU = independent 6-block population, coupled only by grad-avg ->
phase-transition dilution) with FLAT gpu-level: each GPU = one exploration block of ONE global
population; all_gather per-rank rollouts -> reuse augment_batch_for_mixed_expl (num_blocks==ws) ->
equal-shard -> grad all_reduce becomes lossless. Real baseline = full_train_2x_12288.sh
(2 ranks x 12288, 6 blocks/rank). Target = 8x5090, each card = one population.

## Status
- FLAT port DONE (a2c_common +166/-16, custom_utils +14): all_gather_cat, _setup_gpu_level_sapg,
  _enter/_exit_global_sapg_context, _gather_rollout, _shard_batch, train_epoch gpu branch.
  gpu_level_sapg yaml flag, only multi_gpu else byte-identical. py_compile OK. Same diff ported to GMT worktree.
- ruff.toml (extend-exclude rl_games + force-exclude) added to stop global auto-ruff reformatting vendored fork.

## CRITICAL blockers found (adversarial workflow wf_92f922b8, verified vs real SimToolRealSAPG.yaml)
Current flat port only works for MLP + no central_value. Production config CANNOT run it:
- RNN: config has LSTM 1024 before_mlp. _gather_rollout asserts `not self.is_rnn` -> hard fail.
  FIX = gather+shard rnn_states / last_rnn_states / rnn_state_buffer seq/env-aligned; shard on game axis
  without splitting sequences.
- central_value: config has asymmetric central_value_config. _gather_rollout gathers states but
  central_value trains on the SHARDED states with its own buffer -> incoherent. FIX = gather central-value
  inputs coherently + shard consistently.
- advantage-norm (G3): prepare_dataset uses dist_mean_var_count cross-rank; under gpu-level shards are
  slices of ONE global D so it reconstructs global stats but != single-process advantages.std(). FIX to match.
- KL/LR (G4): per-shard KL all_reduce-averaged feeds adaptive LR; != single-process full-D KL. FIX to match.
- (SC2c) off_policy_contrib hardcoded 0 unless env LOG_OFF_POLICY_GRADS=1 -> set it for live checks.

## Fix design
IN PROGRESS via workflow wf_83d63b77 (map data flow -> design diffs -> adversarial verify -> impl spec),
covering RNN, central_value, norm/KL. Apply resulting diffs to BOTH repos, then gate with the
single-process-equivalence test (real network forward: advantages / off-policy returns / rnn / central-value
inputs / KL / per-param grad must equal single-process ws-block run).

## Verification harness (authored by workflow wf_92f922b8, on disk)
- tests/sapg_verify/test_offline_suite.py (65KB, SC1a-SC6b) + run_offline.sh + RUN.md + _common.py.
- tests/test_gpu_level_sapg.py + tests/test_gpu_level_sapg_augment_equiv.py.
- tests/sc5c/ (run_sc5c.sh + sc5c_instrumentation.patch + sc5c_verdict.py) = live 2-GPU efficiency A/B.
Verification tiers: static (veto SC5a preservation, SC6b RNN-scope, SC2c metric-gate) -> CPU-gloo
(veto SC3a cross-rank gather, SC4a bitwise-D-equiv, SC4c lossless-grad) -> live 2-GPU (SC5c efficiency).
KEY upgrade (critique): add single-process<->gathered LEARNING-SIGNAL equivalence with a REAL forward
(not stub) to catch value-recompute / advantage-norm / KL divergence (G1/G3/G4).

## Tests modified
- ADDED tests/test_gpu_level_sapg.py, tests/test_gpu_level_sapg_augment_equiv.py, tests/sapg_verify/*,
  tests/sc5c/* (all new). No existing tests changed.

## Run / Results / Conclusion
(pending fix implementation + cluster verification on dev pod str-isaac-dev; NOT merged/pushed)

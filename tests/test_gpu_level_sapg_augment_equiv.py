"""SC4a: gathered global-D == single-device ws-block SAPG D (bitwise-fp equivalence).

Proves the CORE 'reuse augment unchanged' contract of GPU-level SAPG:
for identical deterministic injected rollouts, the pre-shard aggregated dataset D
produced by the multi-rank path
    per-rank rollout -> _gather_rollout -> _enter_global_sapg_context -> augment_batch_for_mixed_expl
is torch.equal (bitwise fp) to the D produced by ONE process that rolled out
`world_size` blocks of the SAME data and ran the UNCHANGED augment with num_blocks==world_size.

Pure torch + gloo on CPU. No Isaac Sim, no GPU. Run on the cluster dev pod:
    .venv_isaacsim/bin/python -m pytest tests/test_gpu_level_sapg_augment_equiv.py -q

Determinism: repeat_idxs is forced to [0, 1] (removes np.random.choice); get_values is a
fixed linear stub (both paths call the identical fn on identical inputs); no dropout / no BN.
This isolates the AGGREGATE RECONSTRUCTION correctness -- it does NOT exercise sharding,
grad all-reduce, IsaacSim rollout, RNN, or the real network.
"""

import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_games.common.a2c_common import A2CBase

# ---- fixed problem shape (small; layout is what matters, not size) ----------
WS = 2  # world_size == num SAPG blocks after gather
NA = 4  # per-rank num_actors (== per-block env count == intr_coef_block_size)
H = 3  # horizon_length
OBS = 6  # base obs dim (WITHOUT the appended coef embedding)
EMBD = 1  # learn_param/disjoint => embd is 1-D generator value (see _setup, disjoint branch)
ACT = 2  # action dim
GAMMA = 0.99
COEF_SCALE = 0.002
PORT = "tcp://127.0.0.1:29601"

# keys carried in a continuous batch_dict after swap_and_flatten01 (ContinuousA2CBase):
#   update_list = actions, neglogpacs, values, mus, sigmas ; + obses, states, dones ; + returns
# obses/states carry the appended coef-embedding in their LAST EMBD columns.
BATCH_TENSOR_KEYS = [
    "actions",
    "neglogpacs",
    "values",
    "mus",
    "sigmas",
    "obses",
    "states",
    "dones",
    "returns",
]


def _bind(obj, *names):
    for name in names:
        setattr(obj, name, types.MethodType(getattr(A2CBase, name), obj))


def _det_get_values_factory(embd_dim):
    """Deterministic value head: a fixed linear map of obs (incl. the coef-embed cols).
    Both paths use the SAME fn on the SAME inputs, so recomputed follower returns match
    bitwise iff the reconstructed obs/embedding layout matches. Depends on the embed cols
    on purpose (augment relabels them for followers)."""

    def get_values(self, obs_dict, rnn_states=None):
        obs = obs_dict["obs"]
        w = (
            torch.arange(1, obs.shape[-1] + 1, dtype=obs.dtype, device=obs.device)
            * 0.01
        )
        return (obs * w).sum(dim=-1, keepdim=True)  # [N, 1]

    return get_values


def _make_block_rollout(block_id, na, horizon, seed):
    """Deterministic per-block rollout: batch_dict (flattened, env-major dim0) + raw extras
    ([horizon, na, ...] env dim1), matching play_steps output layout for ONE block of na envs.
    Block b's obs/states carry generator value gen[b] in the last EMBD cols (as _setup would set)."""
    g = torch.Generator().manual_seed(1000 + seed)
    N = horizon * na
    gen_all = torch.linspace(50.0, 0.0, WS)  # rank/block generator values
    gval = gen_all[block_id]

    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    # flattened batch_dict (env-major: [horizon*na, ...]); obses last EMBD cols == this block's embd
    obses = r(N, OBS + EMBD)
    obses[:, -EMBD:] = gval
    states = r(N, OBS + EMBD)  # asymmetric critic state; same embed convention
    states[:, -EMBD:] = gval
    batch_dict = {
        "actions": r(N, ACT),
        "neglogpacs": r(N),
        "values": r(N, 1),
        "mus": r(N, ACT),
        "sigmas": torch.abs(r(N, ACT)) + 0.1,
        "obses": obses,
        "states": states,
        "dones": (r(N) > 0.5).float(),
        "returns": r(N, 1),
        "played_frames": horizon * na,
        "step_time": 0.5,
    }
    # raw extras: [horizon, na, ...] (env dim = 1); obs/states last EMBD cols == block embd
    e_obs = r(horizon, na, OBS + EMBD)
    e_obs[:, :, -EMBD:] = gval
    e_states = r(horizon, na, OBS + EMBD)
    e_states[:, :, -EMBD:] = gval
    last_obs = r(na, OBS + EMBD)
    last_obs[:, -EMBD:] = gval
    last_states = r(na, OBS + EMBD)
    last_states[:, -EMBD:] = gval
    extras = {
        "rewards": r(horizon, na, 1),
        "obs": e_obs,
        "states": e_states,
        "dones": (r(horizon, na) > 0.5).float(),
        "last_dones": (r(na) > 0.5).float(),
        "last_obs": {"obs": last_obs, "states": last_states},
        "last_rnn_states": None,
        "rnn_states": None,
        "mb_intr_rewards": None,  # expl_reward_type == entropy => intr_reward_model is None
        "mb_extr_rewards": r(horizon, na, 1),
    }
    return batch_dict, extras


def _configure_common(algo, config):
    """Minimal A2CBase surface needed by _setup_gpu_level_sapg / augment / gather."""
    algo.world_size = WS
    algo.num_actors = NA
    algo.horizon_length = H
    algo.seq_length = H
    algo.gamma = GAMMA
    algo.ppo_device = "cpu"
    algo.is_rnn = False
    algo.expl_type = "mixed_expl_learn_param"  # disjoint/learn_param => 1-D embd
    algo.use_others_experience = "lf"
    algo.intr_reward_model = None
    algo.config = config
    algo.get_values = types.MethodType(_det_get_values_factory(EMBD), algo)
    _bind(
        algo,
        "_setup_gpu_level_sapg",
        "_enter_global_sapg_context",
        "_exit_global_sapg_context",
        "_gather_rollout",
        "augment_batch_for_mixed_expl",
    )


CONFIG = {
    "off_policy_ratio": 1.0,
    "expl_reward_type": "entropy",
    "expl_reward_coef_embd_size": EMBD,
    "expl_reward_coef_scale": COEF_SCALE,
    "expl_coef_block_size": NA,  # unused by gpu-level setup but present in real cfg
}
REPEAT_IDXS = [
    0,
    1,
]  # deterministic: off_policy_ratio=1, num_blocks=ws=2 => only choice


# ---------- reference: ONE process, ws blocks, UNCHANGED augment -------------
def build_single_process_reference():
    algo = types.SimpleNamespace()
    algo.global_rank = 0
    algo.multi_gpu = False  # reference single process: no broadcast in augment
    _configure_common(algo, CONFIG)
    algo._setup_gpu_level_sapg(
        CONFIG
    )  # builds _gpu_global_* (block-ordered ws*na views)
    algo._enter_global_sapg_context()  # num_actors=ws*na, intr_*=global (== upstream ws-block SAPG)

    # roll out ws blocks and concat env-major (dim0) / env dim (dim1) -- EXACTLY the layout
    # all_gather_cat(dim=0)/(dim=1) reconstructs. Block b uses seed b (same as ranks).
    bds, exs = zip(*[_make_block_rollout(b, NA, H, seed=b) for b in range(WS)])
    cat_bd = {}
    for k in BATCH_TENSOR_KEYS:
        cat_bd[k] = torch.cat([bd[k] for bd in bds], dim=0)
    cat_bd["played_frames"] = sum(bd["played_frames"] for bd in bds)
    cat_bd["step_time"] = bds[0]["step_time"]

    cat_ex = {}
    for k in ("rewards", "obs", "states", "dones", "mb_extr_rewards"):
        cat_ex[k] = torch.cat([ex[k] for ex in exs], dim=1)
    cat_ex["last_dones"] = torch.cat([ex["last_dones"] for ex in exs], dim=0)
    cat_ex["last_obs"] = {
        "obs": torch.cat([ex["last_obs"]["obs"] for ex in exs], dim=0),
        "states": torch.cat([ex["last_obs"]["states"] for ex in exs], dim=0),
    }
    cat_ex["last_rnn_states"] = None
    cat_ex["rnn_states"] = None
    cat_ex["mb_intr_rewards"] = None

    D = algo.augment_batch_for_mixed_expl(cat_bd, cat_ex, repeat_idxs=list(REPEAT_IDXS))
    return {k: v for k, v in D.items() if torch.is_tensor(v)}


# ---------- worker: real multi-rank gather + augment (pre-shard) -------------
def _rank_worker(rank, out_file):
    dist.init_process_group("gloo", rank=rank, world_size=WS, init_method=PORT)
    algo = types.SimpleNamespace()
    algo.global_rank = rank
    algo.multi_gpu = (
        True  # real path: augment will dist.broadcast_object_list(repeat_idxs)
    )
    _configure_common(algo, CONFIG)
    algo._setup_gpu_level_sapg(CONFIG)  # local view active (this rank's block only)

    # this rank's own block == seed==rank (same tensors block `rank` gets in the reference)
    bd, ex = _make_block_rollout(rank, NA, H, seed=rank)

    # exact production sequence (train_epoch gpu_level branch, lines 1496-1503):
    algo._enter_global_sapg_context()
    g_bd, g_ex = algo._gather_rollout(bd, ex)
    D = algo.augment_batch_for_mixed_expl(
        g_bd, g_ex, repeat_idxs=list(REPEAT_IDXS)
    )  # pre-shard
    algo._exit_global_sapg_context()

    if rank == 0:
        torch.save({k: v for k, v in D.items() if torch.is_tensor(v)}, out_file)
    # also let rank 1 save so we can assert ranks agree (D identical on every rank)
    if rank == 1:
        torch.save({k: v for k, v in D.items() if torch.is_tensor(v)}, out_file + ".r1")
    dist.barrier()
    dist.destroy_process_group()


def test_sc4a_gathered_D_bitwise_equals_single_process_D(tmp_path):
    out = str(tmp_path / "D_rank0.pt")
    mp.spawn(_rank_worker, args=(out,), nprocs=WS, join=True)
    D_multi_r0 = torch.load(out)
    D_multi_r1 = torch.load(out + ".r1")
    D_single = build_single_process_reference()

    # (a) all ranks produce the SAME pre-shard D (gather makes D identical everywhere)
    assert set(D_multi_r0) == set(D_multi_r1)
    for k in D_multi_r0:
        assert torch.equal(D_multi_r0[k], D_multi_r1[k]), f"ranks disagree on {k}"

    # (b) CORE claim: multi-rank pre-shard D == single-process ws-block D, BITWISE fp
    assert set(D_multi_r0) == set(D_single), (
        f"key mismatch: multi={set(D_multi_r0)} single={set(D_single)}"
    )
    mismatches = []
    for k in sorted(D_single):
        a, b = D_multi_r0[k], D_single[k]
        if a.shape != b.shape:
            mismatches.append(f"{k}: shape {tuple(a.shape)} != {tuple(b.shape)}")
        elif not torch.equal(a, b):  # torch.equal == exact bitwise (no atol/rtol)
            md = (a.float() - b.float()).abs().max().item()
            mismatches.append(f"{k}: max|Δ|={md:.3e}")
    assert not mismatches, "D differs (mechanism broken):\n  " + "\n  ".join(mismatches)

    # sanity: off_policy_mask exists and has BOTH on- and off-policy rows (lf relabel happened)
    m = D_single["off_policy_mask"]
    assert m.dtype == torch.bool and m.any() and (~m).any(), (
        "lf augment produced no off-policy rows"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", "-s"]))

"""Unit tests for GPU-level SAPG (flat: gather + reuse-augment + shard) in SimToolReal's rl_games.

Pure-torch, no Isaac Sim. The distributed transport primitive uses gloo on CPU via mp.spawn.
Run on the cluster dev pod: `.venv_isaacsim/bin/python -m pytest tests/test_gpu_level_sapg.py`.
Locks down the three new pieces: local/global block views, disjoint equal shard partition,
rank-ordered all-gather layout. Full-pipeline equivalence + real multi-GPU run are separate.
"""

import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_games.common.a2c_common import A2CBase
from rl_games.common.custom_utils import all_gather_cat, create_sinusoidal_encoding


def _bind(obj, *names):
    for name in names:
        setattr(obj, name, types.MethodType(getattr(A2CBase, name), obj))


def test_shard_partition_disjoint_equal_union():
    ws, horizon, n_traj = 4, 3, 10  # 10 % 4 != 0 -> 2 tail trajectories dropped
    n = n_traj * horizon
    full = {
        "returns": torch.arange(n).float().reshape(n, 1),
        "obses": torch.arange(n * 5).float().reshape(n, 5),
        "off_policy_mask": torch.zeros(n, dtype=torch.bool),
        "played_frames": 123,
        "step_time": 0.5,
    }
    shards = []
    for r in range(ws):
        algo = types.SimpleNamespace(
            world_size=ws, global_rank=r, horizon_length=horizon, epoch_num=5
        )
        _bind(algo, "_shard_batch")
        shards.append(
            algo._shard_batch(
                {k: (v.clone() if torch.is_tensor(v) else v) for k, v in full.items()}
            )
        )

    sizes = [len(s["returns"]) for s in shards]
    assert len(set(sizes)) == 1, (
        "shards must be equal-sized for lossless grad averaging"
    )
    traj_per = n_traj // ws
    assert sizes[0] == traj_per * horizon
    used = traj_per * ws * horizon
    idx = torch.cat([s["returns"][:, 0] for s in shards]).long()
    assert torch.equal(idx.sort().values, torch.arange(used)), (
        "shards must be a disjoint partition of D[:used]"
    )
    assert shards[0]["played_frames"] == 123


@pytest.mark.parametrize("disjoint", [True, False])
def test_setup_local_matches_global_block(disjoint):
    ws, na, embd_dim = 4, 5, 8
    expl_type = "mixed_expl_learn_param" if disjoint else "mixed_expl"
    ranks = []
    for r in range(ws):
        algo = types.SimpleNamespace(
            world_size=ws,
            global_rank=r,
            num_actors=na,
            ppo_device="cpu",
            expl_type=expl_type,
        )
        _bind(algo, "_setup_gpu_level_sapg")
        algo._setup_gpu_level_sapg(
            {
                "expl_reward_coef_embd_size": embd_dim,
                "expl_reward_type": "entropy",
                "expl_reward_coef_scale": 0.002,
            }
        )
        ranks.append(algo)

    g_embd = ranks[0]._gpu_global_intr_embd
    g_coef = ranks[0]._gpu_global_intr_coef
    for r in range(ws):
        assert torch.allclose(
            ranks[r]._gpu_local_intr_embd, g_embd[r * na : (r + 1) * na]
        )
        assert torch.allclose(
            ranks[r]._gpu_local_intr_coef, g_coef[r * na : (r + 1) * na]
        )

    gen = torch.linspace(50.0, 0.0, ws)
    block_ids = torch.arange(ws).repeat_interleave(na)
    if disjoint:
        exp_embd = gen.reshape(-1, 1)[block_ids]
    else:
        exp_embd = create_sinusoidal_encoding(gen, embd_dim, n=100)[block_ids]
    assert torch.allclose(g_embd, exp_embd)
    exp_coef = (torch.linspace(0.5, 0.0, ws) * 0.002)[block_ids]
    assert torch.allclose(g_coef, exp_coef)


def _agc_worker(rank, ws, dim, out_file):
    dist.init_process_group(
        "gloo", rank=rank, world_size=ws, init_method="tcp://127.0.0.1:29557"
    )
    t = torch.full((2, 3), float(rank))
    g = all_gather_cat(t, ws, dim=dim)
    if rank == 0:
        torch.save(g, out_file)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("dim", [0, 1])
def test_all_gather_cat_rank_order(tmp_path, dim):
    ws = 2
    out = str(tmp_path / f"gathered_{dim}.pt")
    mp.spawn(_agc_worker, args=(ws, dim, out), nprocs=ws, join=True)
    g = torch.load(out)
    if dim == 0:
        assert g.shape == (4, 3)
        assert torch.equal(g[:2], torch.zeros(2, 3))
        assert torch.equal(g[2:], torch.ones(2, 3))
    else:
        assert g.shape == (2, 6)
        assert torch.equal(g[:, :3], torch.zeros(2, 3))
        assert torch.equal(g[:, 3:], torch.ones(2, 3))

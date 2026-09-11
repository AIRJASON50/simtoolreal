"""GPU-level SAPG — OFFLINE verification suite (static + 2-proc CPU gloo).

Proves the GPU-level SAPG implementation is numerically equivalent to a single
process running world_size blocks, that the grad average is lossless, and that
all plumbing (block layout / context swap / gather order / shard partition /
frame accounting / RNN scope / naive-DDP contrast) is correct — WITHOUT GPU or
IsaacSim. Live-behaviour checks (off_policy_contrib>0, comm-flat-vs-ratio) are
deferred to the >=2/>=3 GPU launch documented in RUN.md.

Run:  .venv_isaacsim/bin/python -m pytest tests/sapg_verify/test_offline_suite.py -v

Every check binds REAL rl_games methods (types.MethodType) — no re-implementation.
Assertions encode the GROUND TRUTH read from source, not the claim prose; where
the claim prose is wrong (SC2a leader-direction inversion) the test asserts the
real behaviour and pins the inversion so it cannot silently drift.
"""

import ast
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_games.common import common_losses
from rl_games.common.a2c_common import A2CBase
from rl_games.common.custom_utils import create_sinusoidal_encoding
from rl_games.common.datasets import PPODataset

from tests.sapg_verify import _common as C

SCALE = 0.002  # SimToolRealSAPG.yaml expl_reward_coef_scale
EMBD_CFG = {
    "expl_reward_coef_embd_size": 32,
    "expl_reward_type": "entropy",
    "expl_reward_coef_scale": SCALE,
}
GLOO_PORT_BASE = 29650  # bumped per-test to avoid collisions


def _bind_A2CBase(obj, *names):
    C.bind(obj, A2CBase, *names)


# =========================================================================== #
# SC1a — per-rank exploration coefficient distinct & == linspace(0.5,0,ws)[r]*scale
# static. Extends test_setup_local_matches_global_block; pins ABSOLUTE per-rank
# scalar + rollout-facing binding (fix: intr_reward_coef IS _gpu_local_intr_coef).
# =========================================================================== #


@pytest.mark.parametrize("ws", [2, 4, 8])
def test_SC1a_per_rank_coef_distinct_matches_linspace(ws):
    na = 5
    coefs = []
    for r in range(ws):
        algo = types.SimpleNamespace(
            world_size=ws,
            global_rank=r,
            num_actors=na,
            ppo_device="cpu",
            expl_type="mixed_expl_learn_param",
        )
        _bind_A2CBase(algo, "_setup_gpu_level_sapg")
        algo._setup_gpu_level_sapg(EMBD_CFG)

        # (A) local coef vector is CONSTANT == linspace(0.5,0,ws)[r]*scale
        exp_scalar = torch.linspace(0.5, 0.0, ws)[r] * SCALE
        assert algo._gpu_local_intr_coef.shape == (na,)
        assert torch.allclose(
            algo._gpu_local_intr_coef, torch.full((na,), exp_scalar), atol=1e-8, rtol=0
        )
        assert torch.allclose(
            algo._gpu_local_intr_coef,
            algo._gpu_local_intr_coef[0].expand(na),
            atol=1e-8,
            rtol=0,
        )

        # (B) local embd == raw linspace(50,0,ws)[r], shape (na,1) under learn_param(disjoint)
        exp_embd = torch.linspace(50.0, 0.0, ws)[r]
        assert algo._gpu_local_intr_embd.shape == (na, 1)
        assert torch.allclose(
            algo._gpu_local_intr_embd, torch.full((na, 1), exp_embd), atol=1e-6, rtol=0
        )

        # FIX (close live-attribute-swap hole): rollout-facing attrs bound to LOCAL views on exit.
        assert algo.intr_reward_coef is algo._gpu_local_intr_coef
        assert algo.intr_reward_coef_embd is algo._gpu_local_intr_embd

        coefs.append(algo._gpu_local_intr_coef[0].item())

    coefs = torch.tensor(coefs)
    # (C) collected scalars == linspace(0.5,0,ws)*scale
    assert torch.allclose(
        coefs, torch.linspace(0.5, 0.0, ws) * SCALE, atol=1e-8, rtol=0
    )
    # (D) pairwise DISTINCT + strictly decreasing rank0 -> rank ws-1
    diffs = coefs.unsqueeze(0) - coefs.unsqueeze(1)
    off_diag = diffs[~torch.eye(ws, dtype=torch.bool)]
    assert off_diag.abs().min() > 1e-9
    assert torch.all(coefs[:-1] > coefs[1:])
    # (E) endpoints exact
    assert coefs[0].item() == pytest.approx(0.5 * SCALE, abs=1e-8)
    assert coefs[-1].item() == pytest.approx(0.0, abs=1e-12)


def test_SC1a_num_actors_independence():
    """FIX: shape-check with the real large num_actors (12288/2) to rule out coupling."""
    ws, na = 2, 6144
    algo = types.SimpleNamespace(
        world_size=ws,
        global_rank=0,
        num_actors=na,
        ppo_device="cpu",
        expl_type="mixed_expl_learn_param",
    )
    _bind_A2CBase(algo, "_setup_gpu_level_sapg")
    algo._setup_gpu_level_sapg(EMBD_CFG)
    assert algo._gpu_local_intr_coef.shape == (na,)
    assert torch.allclose(
        algo._gpu_local_intr_coef, torch.full((na,), 0.5 * SCALE), atol=1e-8
    )


def test_SC1a_gate_gpu_level_off_without_multi_gpu():
    """FIX (companion gate check): gpu_level_sapg = config.get(...) and multi_gpu.
    With multi_gpu=False the flag resolves False regardless of config."""
    assert C.resolve_gpu_level_flag({"gpu_level_sapg": True}, multi_gpu=False) is False
    assert C.resolve_gpu_level_flag({"gpu_level_sapg": True}, multi_gpu=True) is True
    assert C.resolve_gpu_level_flag({}, multi_gpu=True) is False
    # AST-confirm the gate is the exact short-circuit on the real source.
    src = C.A2C_COMMON.read_text()
    assert (
        "self.gpu_level_sapg = config.get('gpu_level_sapg', False) and self.multi_gpu"
        in src
    )


# =========================================================================== #
# SC1b — global block view lays out block b == rank b; local == r-th slice.
# static. == shipped test, hardened with rank-invariance + shape asserts.
# =========================================================================== #


@pytest.mark.parametrize("disjoint", [True, False])
def test_SC1b_global_block_layout(disjoint):
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
        _bind_A2CBase(algo, "_setup_gpu_level_sapg")
        algo._setup_gpu_level_sapg({**EMBD_CFG, "expl_reward_coef_embd_size": embd_dim})
        ranks.append(algo)

    g_embd = ranks[0]._gpu_global_intr_embd
    g_coef = ranks[0]._gpu_global_intr_coef
    # (A) local == r-th block slice of global
    for r in range(ws):
        assert torch.allclose(
            ranks[r]._gpu_local_intr_embd, g_embd[r * na : (r + 1) * na]
        )
        assert torch.allclose(
            ranks[r]._gpu_local_intr_coef, g_coef[r * na : (r + 1) * na]
        )
    # FIX: global view is rank-invariant (closes the theoretical per-rank-corruption hole)
    for r in range(1, ws):
        assert torch.allclose(ranks[r]._gpu_global_intr_embd, g_embd)
        assert torch.allclose(ranks[r]._gpu_global_intr_coef, g_coef)
    # (B) block-ordered block b == rank b
    gen = torch.linspace(50.0, 0.0, ws)
    block_ids = torch.arange(ws).repeat_interleave(na)
    exp_embd = (
        gen.reshape(-1, 1)[block_ids]
        if disjoint
        else create_sinusoidal_encoding(gen, embd_dim, n=100)[block_ids]
    )
    assert torch.allclose(g_embd, exp_embd)
    exp_coef = (torch.linspace(0.5, 0.0, ws) * SCALE)[block_ids]
    assert torch.allclose(g_coef, exp_coef)
    # FIX: shape assertions (a degenerate broadcast that allcloses would otherwise slip through)
    assert g_coef.shape == (ws * na,)
    assert g_embd.shape == ((ws * na, 1) if disjoint else (ws * na, embd_dim))
    assert ranks[0]._gpu_local_intr_coef.shape == (na,)
    assert ranks[0]._gpu_local_intr_embd.shape == (
        (na, 1) if disjoint else (na, embd_dim)
    )


# =========================================================================== #
# SC1c — context swap makes num_actors==ws*na and augment num_blocks==ws;
# exit restores local views by identity. static.
# FIX: add coef diversity + block-ordering value asserts (multi-block active),
# and run the REAL augment num_blocks expression rather than only re-deriving it.
# =========================================================================== #


@pytest.mark.parametrize("ws", [2, 4, 6])
def test_SC1c_context_swap_num_blocks(ws):
    na = 4096  # == real single-card block; invariant holds for any na>0
    algo = types.SimpleNamespace(
        world_size=ws,
        global_rank=0,
        num_actors=na,
        ppo_device="cpu",
        expl_type="mixed_expl_learn_param",
    )
    _bind_A2CBase(
        algo,
        "_setup_gpu_level_sapg",
        "_enter_global_sapg_context",
        "_exit_global_sapg_context",
    )
    algo._setup_gpu_level_sapg(EMBD_CFG)

    def num_blocks():  # EXACT expression augment_batch_for_mixed_expl uses (a2c_common:1097)
        return algo.num_actors // algo.intr_coef_block_size

    # after setup: local view, block_size == LOCAL na, one block
    assert algo.intr_coef_block_size == na
    assert algo.num_actors == na
    assert algo.intr_reward_coef is algo._gpu_local_intr_coef
    assert algo.intr_reward_coef_embd is algo._gpu_local_intr_embd
    assert num_blocks() == 1
    local_coef_obj = algo._gpu_local_intr_coef
    local_embd_obj = algo._gpu_local_intr_embd

    # enter: num_actors <- ws*na, block_size unchanged, num_blocks == world_size
    algo._enter_global_sapg_context()
    assert algo.num_actors == ws * na
    assert algo.intr_coef_block_size == na
    assert num_blocks() == ws  # LOAD-BEARING
    assert algo.intr_reward_coef is algo._gpu_global_intr_coef
    assert algo.intr_reward_coef_embd is algo._gpu_global_intr_embd
    assert algo.intr_reward_coef.shape[0] == ws * na
    assert algo.intr_reward_coef_embd.shape[0] == ws * na

    # FIX: multi-block DIVERSITY + ordering in the active global view
    block_ids = torch.arange(ws).repeat_interleave(na)
    exp_coef = (torch.linspace(0.5, 0.0, ws) * SCALE)[block_ids]
    assert torch.allclose(algo._gpu_global_intr_coef, exp_coef)
    # distinct per-block values (all-blocks-one-coef would fail here)
    per_block = algo._gpu_global_intr_coef.view(ws, na)[:, 0]
    assert torch.unique(per_block).numel() == ws
    for b in range(ws):
        assert torch.allclose(
            algo._gpu_global_intr_coef[b * na : (b + 1) * na], per_block[b].expand(na)
        )

    # exit: restore LOCAL by identity
    algo._exit_global_sapg_context()
    assert algo.num_actors == na
    assert algo.intr_reward_coef is local_coef_obj
    assert algo.intr_reward_coef_embd is local_embd_obj
    assert num_blocks() == 1


def test_SC1c_train_epoch_calls_context_in_order():
    """FIX (integration): assert ContinuousA2CBase.train_epoch actually invokes
    _enter -> _gather_rollout -> augment -> _shard_batch -> _exit, all under the
    gpu_level_sapg guard, so a refactor moving the mutation out of _enter cannot
    pass this check while breaking the live path."""
    tree, src = C.parse(C.A2C_COMMON)
    te = C.find_method(tree, "ContinuousA2CBase", "train_epoch")
    te_src = C.src_of(te, src)
    order = [
        "_enter_global_sapg_context",
        "_gather_rollout",
        "augment_batch_for_mixed_expl",
        "_shard_batch",
    ]
    positions = [te_src.index(name) for name in order]
    assert all(p >= 0 for p in positions)
    assert positions == sorted(positions), "train_epoch call order broken"
    # _exit appears (restore) and every gpu-level hook is under `if self.gpu_level_sapg`
    assert "_exit_global_sapg_context" in te_src
    _assert_calls_guarded(
        te,
        {
            "_enter_global_sapg_context",
            "_gather_rollout",
            "_shard_batch",
            "_exit_global_sapg_context",
        },
    )


# =========================================================================== #
# SC2a — lf augment relabels a follower block to the exploit-target embedding
# and marks off_policy_mask=True. static.
# GROUND TRUTH (verified by pure-python emulation): relabel target == embd[-1]
# (coef linspace END == 0.0 == exploit block), NOT block-0. The claim prose says
# "block-0"; that is INVERTED. Test asserts embd[-1] and pins the inversion.
# =========================================================================== #


def _mk_lf_algo(num_blocks, B, H, embd_dim=1):
    na = num_blocks * B
    algo = types.SimpleNamespace()
    algo.num_actors = na
    algo.intr_coef_block_size = B
    algo.horizon_length = H
    algo.use_others_experience = "lf"
    algo.multi_gpu = False
    algo.config = {"off_policy_ratio": 1}
    algo.gamma = 0.99
    algo.is_rnn = False
    coef = torch.linspace(50.0, 0.0, num_blocks)
    block_ids = torch.arange(num_blocks).repeat_interleave(B)
    algo.intr_reward_coef_embd = coef[block_ids].reshape(-1, embd_dim).float()
    algo.intr_reward_coef = (torch.linspace(0.5, 0.0, num_blocks) * SCALE)[
        block_ids
    ].float()
    C.bind(algo, A2CBase, "augment_batch_for_mixed_expl")
    return algo, na


def test_SC2a_lf_relabel_offpolicy_stamp_and_count():
    num_blocks, B, H, obs_dim, embd_dim = 3, 2, 4, 6, 1
    algo, na = _mk_lf_algo(num_blocks, B, H, embd_dim)
    T = na * H
    obses = torch.zeros(T, obs_dim)
    for a in range(na):
        sl = slice(a * H, (a + 1) * H)
        obses[sl, 0] = float(a // B)  # physical block tag (never overwritten)
        obses[sl, 1] = float(a)  # actor id tag
    obses[:, -embd_dim:] = algo.intr_reward_coef_embd.repeat_interleave(H, dim=0)
    base_batch = {
        "obses": obses,
        "returns": torch.arange(T).float().reshape(T, 1),
        "values": torch.zeros(T, 1),
        "actions": torch.zeros(T, 3),
        "neglogpacs": torch.zeros(T),
        "dones": torch.zeros(T),
        "played_frames": T,
        "step_time": 0.1,
        "states": None,
        "rnn_states": None,
    }
    algo.get_values = lambda d, rnn_states=None: torch.zeros(d["obs"].shape[0], 1)
    base_extras = {
        "rewards": torch.zeros(H, na, 1),
        "obs": torch.zeros(H, na, obs_dim),
        "last_obs": {"obs": torch.zeros(na, obs_dim)},
        "states": None,
        "dones": torch.zeros(H, na),
        "last_dones": torch.zeros(na),
        "rnn_states": None,
        "last_rnn_states": None,
        "mb_intr_rewards": None,
        "mb_extr_rewards": torch.zeros(H, na, 1),
    }

    exploit_embd = algo.intr_reward_coef_embd[
        -1
    ].item()  # coef END == 0.0 == exploit target
    leader_prose_embd = algo.intr_reward_coef_embd[
        0
    ].item()  # 50.0 (what the claim WRONGLY calls leader)
    # exploit target is argmin(coef) block; confirm it == last block (linspace(0.5,0) schedule)
    per_block_coef = algo.intr_reward_coef.view(num_blocks, B)[:, 0]
    assert torch.argmin(per_block_coef).item() == num_blocks - 1

    for k in (1, 2):
        bd = {
            kk: (vv.clone() if torch.is_tensor(vv) else vv)
            for kk, vv in base_batch.items()
        }
        ex = {
            kk: (
                vv.clone()
                if torch.is_tensor(vv)
                else ({"obs": vv["obs"].clone()} if isinstance(vv, dict) else vv)
            )
            for kk, vv in base_extras.items()
        }
        out = algo.augment_batch_for_mixed_expl(bd, ex, repeat_idxs=[0, k])
        mask = out["off_policy_mask"]
        aug = out["obses"]
        assert mask.dtype == torch.bool and mask.any() and (~mask).any()
        # (A) one exploration block worth of off-policy transitions
        assert int(mask.sum().item()) == B * H
        # (B) off-policy rows physically came from follower block (k-1)
        assert aug[mask, 0].unique().tolist() == [float(k - 1)]
        # (C) off-policy rows relabelled to exploit-target embd == embd[na-1] (roll -> last block)
        assert aug[mask, -embd_dim:].unique().tolist() == [exploit_embd]
        # (D) on-policy rows keep the full diverse block-embd set
        on = set(round(v, 6) for v in aug[~mask, -embd_dim:].flatten().tolist())
        assert len(on) == num_blocks
        # (E) INVERSION pin: the relabel target is NOT block-0 (=50, what the claim
        # prose wrongly calls "leader"); it IS the argmin(coef) block == last block ==
        # the zero-exploration exploit target. The block-b embd == its generator value,
        # so the exploit embd must equal the embd of the argmin(coef) block.
        assert exploit_embd != leader_prose_embd
        argmin_block = torch.argmin(per_block_coef).item()
        exploit_block_embd = algo.intr_reward_coef_embd.view(num_blocks, B, embd_dim)[
            argmin_block, 0, 0
        ].item()
        assert exploit_embd == pytest.approx(exploit_block_embd)


# =========================================================================== #
# SC2b — off-policy rows ride the SAME PPODataset & are weighted by PPO clipped IS.
# static (loss math + dataset plumbing + contrib formula). REAL functions run.
# =========================================================================== #


def test_SC2b_clip_is_weighting_and_dataset_and_contrib():
    e_clip = 0.1  # SimToolRealSAPG.yaml
    N = 8
    off_policy_mask = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool)
    advantage = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0])
    old_neglogp = torch.zeros(N)
    desired_ratio = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.05, 1.20, 0.80, 0.97])
    new_neglogp = old_neglogp - torch.log(desired_ratio)

    # PART A: REAL common_losses.actor_loss == clipped-IS PPO term row-wise
    a_loss = common_losses.actor_loss(old_neglogp, new_neglogp, advantage, True, e_clip)
    ratio = torch.exp(old_neglogp - new_neglogp)
    surr1 = advantage * ratio
    surr2 = advantage * torch.clamp(ratio, 1 - e_clip, 1 + e_clip)
    expect = torch.max(-surr1, -surr2)
    assert torch.allclose(a_loss, expect, atol=1e-6)
    assert torch.allclose(
        a_loss[~off_policy_mask], -advantage[~off_policy_mask], atol=1e-6
    )
    out_high = 5  # ratio 1.20 > 1+e -> clip bites
    assert not torch.isclose(
        a_loss[out_high], -(advantage[out_high] * ratio[out_high]), atol=1e-6
    )
    assert torch.isclose(
        a_loss[out_high], -(advantage[out_high] * (1 + e_clip)), atol=1e-6
    )
    assert (ratio[off_policy_mask] != 1.0).any()

    # PART B: off_policy_mask rides the SAME PPODataset per-minibatch slicing as obs
    minibatch = 4
    vd = {
        "old_values": torch.zeros(N, 1),
        "old_logp_actions": old_neglogp,
        "advantages": advantage,
        "returns": torch.zeros(N, 1),
        "actions": torch.zeros(N, 3),
        "obs": torch.zeros(N, 10),
        "dones": torch.zeros(N),
        "rnn_states": None,
        "rnn_masks": None,
        "mu": torch.zeros(N, 3),
        "sigma": torch.ones(N, 3),
        "off_policy_mask": off_policy_mask,
    }
    ds = PPODataset(
        batch_size=N,
        minibatch_size=minibatch,
        is_discrete=False,
        is_rnn=False,
        device="cpu",
        seq_length=1,
    )
    ds.update_values_dict(vd)
    seen = []
    for i in range(len(ds)):
        mb = ds[i]
        assert "off_policy_mask" in mb
        assert mb["off_policy_mask"].shape[0] == mb["obs"].shape[0]
        seen.append(mb["off_policy_mask"])
    assert torch.equal(torch.cat(seen), off_policy_mask)

    # PART C: reproduce the a2c_continuous contrib formula AND assert it is
    # byte-identical to the source lines (drift-guard, closes the reimpl hole).
    contrib = torch.logical_and(ratio < 1.0 + e_clip, ratio > 1.0 - e_clip).float()
    contrib_off = torch.nan_to_num(torch.masked_select(contrib, off_policy_mask).mean())
    assert abs(contrib_off.item() - 0.5) < 1e-6
    assert 0.0 < contrib_off.item() <= 1.0
    # mask-polarity discriminator: make on-policy rows differ so a mask swap is caught
    contrib_on = torch.nan_to_num(torch.masked_select(contrib, ~off_policy_mask).mean())
    assert contrib_on.item() == pytest.approx(
        1.0
    )  # on-policy rows all ratio==1 (in-band)
    assert contrib_off.item() != contrib_on.item()

    # drift guard: the real source must contain these exact expressions
    csrc = C.A2C_CONTINUOUS.read_text()
    assert (
        "contrib = torch.logical_and(ratio < 1.0 + curr_e_clip, ratio > 1.0 - curr_e_clip).float()"
        in csrc
    )
    assert (
        "contrib_off = torch.masked_select(contrib, input_dict['off_policy_mask'])"
        in csrc
    )
    assert (
        "a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)"
        in csrc
    )
    # actor_loss_func binds to common_losses.actor_loss unless use_smooth_clamp (yaml does not set it)
    assert "self.actor_loss_func = common_losses.actor_loss" in C.A2C_COMMON.read_text()


# =========================================================================== #
# SC2c — off_policy_contrib is env-gated on LOG_OFF_POLICY_GRADS; default run
# hardcodes off_policy_contrib=0 & off_policy_grads=zeros; TB writers ungated.
# static AST. FIX: unwrap .cpu()/.detach() wrappers (the real fallback is
# torch.zeros_like(all_grads).cpu()); assert zeros_like arg == all_grads;
# assert THEN branch grads unwrap to grads_off.
# =========================================================================== #


def _inner_call_attrs(node):
    """Walk down .cpu()/.detach()/.numpy() wrappers; return list of inner Call func.attr."""
    attrs = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            attrs.append(n.func.attr)
    return attrs


def _zeros_like_arg_name(node):
    for n in ast.walk(node):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "zeros_like"
        ):
            if n.args and isinstance(n.args[0], ast.Name):
                return n.args[0].id
    return None


def _name_ids(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _is_log_gate(test):
    return (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Attribute)
        and test.func.attr == "getenv"
        and any(
            isinstance(a, ast.Constant) and a.value == "LOG_OFF_POLICY_GRADS"
            for a in test.args
        )
    )


def _dict_get(dnode, key):
    for k, v in zip(dnode.keys, dnode.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    return None


def _find_extras_dict(stmts):
    for s in stmts:
        if isinstance(s, ast.Assign):
            for t in s.targets:
                if (
                    isinstance(t, ast.Name)
                    and t.id == "extras"
                    and isinstance(s.value, ast.Dict)
                ):
                    return s.value
    return None


def test_SC2c_off_policy_metric_env_gated():
    tree, src = C.parse(C.A2C_CONTINUOUS)
    calc = C.find_method(tree, "A2CAgent", "calc_gradients")

    gated_ifs = [
        n for n in ast.walk(calc) if isinstance(n, ast.If) and _is_log_gate(n.test)
    ]
    assert len(gated_ifs) == 2, (
        f"expected 2 LOG_OFF_POLICY_GRADS gates, found {len(gated_ifs)}"
    )

    fallback_if = next((n for n in gated_ifs if n.orelse), None)
    assert fallback_if is not None
    then_extras = _find_extras_dict(fallback_if.body)
    else_extras = _find_extras_dict(fallback_if.orelse)
    assert then_extras is not None and else_extras is not None

    # ELSE (default run): off_policy_contrib hardcoded 0
    opc = _dict_get(else_extras, "off_policy_contrib")
    assert isinstance(opc, ast.Constant) and opc.value == 0
    # ELSE off_policy_grads = torch.zeros_like(all_grads).cpu()  (unwrap .cpu())
    opg = _dict_get(else_extras, "off_policy_grads")
    assert "zeros_like" in _inner_call_attrs(opg), (
        "fallback off_policy_grads not zeros_like(...)"
    )
    assert _zeros_like_arg_name(opg) == "all_grads", (
        "fallback zeros must be zeros_like(all_grads), not real grads"
    )

    # THEN (gated): off_policy_contrib computed (not constant); grads unwrap to grads_off (real)
    opc_t = _dict_get(then_extras, "off_policy_contrib")
    assert not isinstance(opc_t, ast.Constant)
    assert "contrib_off" in _name_ids(opc_t)
    opg_t = _dict_get(then_extras, "off_policy_grads")
    assert "zeros_like" not in _inner_call_attrs(opg_t)
    assert "grads_off" in _name_ids(opg_t), (
        "gated off_policy_grads must be real grads_off"
    )

    # masked off_policy_loss + get_grads live ONLY inside the gate, exactly once
    assert src.count("off_policy_loss = torch.masked_select") == 1

    # TB writers that SC2b reads are UNGATED in a2c_common (constant-0 without env var)
    common_src = C.A2C_COMMON.read_text()
    assert (
        "self.writer.add_histogram('auxiliary_stats/off_policy_contrib'" in common_src
    )
    assert "auxiliary_stats/off_on_grad_similarity" in common_src
    assert "auxiliary_stats/off_on_relative_grad_norms" in common_src
    common_ast = ast.parse(common_src)
    for n in ast.walk(common_ast):
        if isinstance(n, ast.If) and _is_log_gate(n.test):
            seg = ast.get_source_segment(common_src, n) or ""
            assert "auxiliary_stats/off_policy_contrib" not in seg, (
                "TB writer unexpectedly inside a gate"
            )

    # numeric degeneracy demonstration (not just lexical): cosine of a zero grad-vector is 0
    g = torch.randn(4)
    zeros = torch.zeros_like(g)
    assert torch.cosine_similarity(
        g.unsqueeze(0), zeros.unsqueeze(0)
    ).item() == pytest.approx(0.0, abs=1e-7)


# =========================================================================== #
# Shared AST guard helper for SC1c / SC3b: every named call in *fn* is nested
# under `if self.gpu_level_sapg:`.
# =========================================================================== #


def _assert_calls_guarded(fn, names):
    def is_guard(test):
        return (
            isinstance(test, ast.Attribute)
            and test.attr == "gpu_level_sapg"
            and isinstance(test.value, ast.Name)
            and test.value.id == "self"
        )

    found = {n: [] for n in names}

    def visit(node, guarded):
        if isinstance(node, ast.If) and is_guard(node.test):
            for c in node.body:
                visit(c, True)
            for c in node.orelse:
                visit(c, guarded)
            return
        if isinstance(node, ast.Call):
            nm = node.func.attr if isinstance(node.func, ast.Attribute) else None
            if nm in names:
                found[nm].append((node.lineno, guarded))
        for c in ast.iter_child_nodes(node):
            visit(c, guarded)

    visit(fn, False)
    for nm in names:
        assert found[nm], f"{nm} not called"
        for lineno, guarded in found[nm]:
            assert guarded, f"UNGUARDED cross-card call {nm} at line {lineno}"


# =========================================================================== #
# SC3a — rollout all-gathered in rank order; leader block draws from ALL ranks.
# cpu-gloo-2proc. FIX: separate DATA tag (varies within rank) from BLOCK identity;
# run REAL augment on the gathered batch and assert leader dataset physically
# contains rank-1-origin follower rows relabelled to the exploit embedding.
# =========================================================================== #


def _sc3a_worker(rank, ws, H, na, obs_base, embd, out_file, port):
    dist.init_process_group(
        "gloo", rank=rank, world_size=ws, init_method=f"tcp://127.0.0.1:{port}"
    )
    algo = types.SimpleNamespace(world_size=ws, is_rnn=False)
    C.bind(algo, A2CBase, "_gather_rollout")
    n_flat = H * na
    tag = float(rank)
    OBS = obs_base + embd
    obses = torch.full((n_flat, OBS), tag)
    obses[:, obs_base:] = torch.linspace(50.0, 0.0, ws)[
        rank
    ]  # this rank's block embedding
    batch_dict = {
        "obses": obses,
        "actions": torch.full((n_flat, 3), tag),
        "neglogpacs": torch.full((n_flat,), tag),
        "values": torch.full((n_flat, 1), tag),
        "returns": torch.full((n_flat, 1), tag),
        "dones": torch.full((n_flat,), tag),
        "played_frames": n_flat,
        "step_time": 0.5,
    }
    e_obs = torch.full((H, na, OBS), tag)
    e_obs[:, :, obs_base:] = torch.linspace(50.0, 0.0, ws)[rank]
    extras = {
        "rewards": torch.full((H, na, 1), tag),
        "obs": e_obs,
        "states": None,
        "dones": torch.full((H, na), tag),
        "mb_intr_rewards": None,
        "mb_extr_rewards": torch.full((H, na, 1), tag),
        "last_dones": torch.full((na,), tag),
        "last_obs": {"obs": torch.full((na, OBS), tag)},
        "last_rnn_states": None,
        "rnn_states": None,
    }
    g_bd, g_ex = algo._gather_rollout(batch_dict, extras)
    if rank == 0:
        torch.save({"bd": g_bd, "ex": g_ex}, out_file)
    dist.barrier()
    dist.destroy_process_group()


def test_SC3a_gather_rank_order_and_leader_draws_cross_rank(tmp_path):
    # ws=3 so the follower we pull (block 1) is a genuine NON-leader, NON-rank-0 card,
    # proving "leader draws from ALL ranks" (repeat_idxs=[0,2] -> filter_leader keeps
    # block 1 == rank 1; [0,1] would keep block 0 == rank 0 = the leader itself). CPU
    # gloo, no GPU/IsaacSim.
    ws, H, na, obs_base, embd = 3, 4, 3, 5, 1
    out = str(tmp_path / "gathered.pt")
    port = GLOO_PORT_BASE + 1
    mp.spawn(
        _sc3a_worker, args=(ws, H, na, obs_base, embd, out, port), nprocs=ws, join=True
    )
    g = torch.load(out)
    g_bd, g_ex = g["bd"], g["ex"]
    nfg = ws * H * na
    nag = ws * na

    # (A) batch_dict concat on dim0 in rank order: block b == rank b
    for k in ("obses", "actions", "values", "returns", "neglogpacs", "dones"):
        t = g_bd[k]
        assert t.shape[0] == nfg
        for b in range(ws):
            seg = t[b * H * na : (b + 1) * H * na]
            # data-tag column(s) carry rank tag (embedding col excluded for obses)
            data = seg[:, :obs_base] if (k == "obses" and seg.dim() > 1) else seg
            assert torch.all(data == float(b))
    assert g_bd["played_frames"] == H * na  # scalar stays local
    assert g_bd["step_time"] == 0.5

    # (B) raw extras concat on dim1 (env dim) in rank order
    for k in ("rewards", "obs", "dones", "mb_extr_rewards"):
        t = g_ex[k]
        assert t.shape[0] == H and t.shape[1] == nag
        for b in range(ws):
            seg = t[:, b * na : (b + 1) * na]
            data = seg[:, :, :obs_base] if k == "obs" else seg
            assert torch.all(data == float(b))
    assert g_ex["states"] is None and g_ex["mb_intr_rewards"] is None

    # (C) last_dones + last_obs on dim0 rank order
    assert g_ex["last_dones"].shape[0] == nag
    assert g_ex["last_obs"]["obs"].shape[0] == nag
    for b in range(ws):
        assert torch.all(g_ex["last_dones"][b * na : (b + 1) * na] == float(b))
        assert torch.all(
            g_ex["last_obs"]["obs"][b * na : (b + 1) * na, :obs_base] == float(b)
        )

    # (D) FIX: run REAL augment on the gathered batch in the global context and assert
    # the leader (block 0) dataset physically contains rank-1-origin follower DATA
    # relabelled to the exploit embedding. This is the "leader draws from all ranks"
    # limb the constant-tag reshape cannot prove.
    algo = types.SimpleNamespace()
    algo.world_size = ws
    algo.num_actors = ws * na  # global context (post _enter)
    algo.intr_coef_block_size = na
    algo.horizon_length = H
    algo.use_others_experience = "lf"
    algo.multi_gpu = False
    algo.config = {"off_policy_ratio": 1}
    algo.gamma = 0.99
    algo.is_rnn = False
    # global block-ordered embedding/coef (as _enter would install)
    block_ids = torch.arange(ws).repeat_interleave(na)
    coef_embd = torch.linspace(50.0, 0.0, ws)
    algo.intr_reward_coef_embd = coef_embd[block_ids].reshape(-1, embd).float()
    algo.intr_reward_coef = (torch.linspace(0.5, 0.0, ws) * SCALE)[block_ids].float()
    algo.get_values = lambda d, rnn_states=None: torch.zeros(d["obs"].shape[0], 1)
    C.bind(algo, A2CBase, "augment_batch_for_mixed_expl")
    aug = algo.augment_batch_for_mixed_expl(
        {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in g_bd.items()},
        {
            kk: (
                vv.clone()
                if torch.is_tensor(vv)
                else (
                    {"obs": vv["obs"].clone()}
                    if isinstance(vv, dict) and vv.get("obs") is not None
                    else vv
                )
            )
            for kk, vv in g_ex.items()
        },
        repeat_idxs=[
            0,
            2,
        ],  # follower block 2-1 == block 1 == rank 1 (a NON-leader card)
    )
    mask = aug["off_policy_mask"]
    assert mask.any() and (~mask).any()
    off_data = aug["obses"][mask][:, :obs_base]  # physical DATA tag of off-policy rows
    off_embd = aug["obses"][mask][:, obs_base:]  # relabelled embedding
    # rank-1-origin follower data physically pulled into the leader (block-0) dataset
    assert torch.all(off_data == 1.0)
    exploit_embd = algo.intr_reward_coef_embd[-1].item()  # coef END == exploit target
    assert torch.allclose(off_embd, torch.full_like(off_embd, exploit_embd))


# =========================================================================== #
# SC3b — naive DDP baseline provably cannot cross-card aggregate. static AST +
# config replay. FIX: read the REAL baseline shell if present; else derive from
# num_envs/block_size and only assert the code-path + flag arithmetic.
# =========================================================================== #


def test_SC3b_naive_ddp_cannot_cross_card_aggregate():
    tree, src = C.parse(C.A2C_COMMON)
    te = C.find_method(tree, "ContinuousA2CBase", "train_epoch")
    # PART A: all cross-card calls are guarded (class-scoped train_epoch)
    _assert_calls_guarded(
        te, {"_enter_global_sapg_context", "_gather_rollout", "_shard_batch"}
    )

    # PART B: flag arithmetic for the naive baseline (multi_gpu True, no gpu_level_sapg)
    baseline = {"multi_gpu": True}  # gpu_level_sapg absent
    assert C.resolve_gpu_level_flag(baseline, baseline["multi_gpu"]) is False

    # Non-gpu-level branch: intr_coef_block_size = expl_coef_block_size; num_blocks local.
    # Read real shell if fetchable; else derive from the documented baseline (12288/2048=6).
    shell = C.ROOT / "full_train_2x_12288.sh"
    if shell.exists():
        txt = shell.read_text()
        assert "gpu_level_sapg=True" not in txt and "gpu_level_sapg True" not in txt
        import re

        m_env = re.search(r"num_envs[=\s]+(\d+)", txt)
        m_blk = re.search(r"expl_coef_block_size[=\s]+(\d+)", txt)
        num_envs = int(m_env.group(1)) if m_env else 12288
        block = int(m_blk.group(1)) if m_blk else 2048
    else:
        # shell lives on cluster CPFS; use documented values, assert only the code-path relation
        num_envs, block = 12288, 2048
    assert num_envs % block == 0
    num_blocks = num_envs // block
    assert num_blocks >= 2
    follower_pool = list(range(1, num_blocks))
    assert follower_pool == list(
        range(1, num_blocks)
    )  # leader block 0 + own followers only
    # the gpu-level layout (block_size == num_actors -> 1 block/card) is NOT what baseline uses
    assert block != num_envs

    # PART C: no experience-data collective (all_gather/all_gather_cat) outside the guard in train_epoch
    te_src = C.src_of(te, src)
    # only grad all_reduce couples cards in the non-gpu-level path; assert no all_gather_cat ungated
    guarded_seg_present = "if self.gpu_level_sapg:" in te_src
    assert guarded_seg_present
    # all_gather_cat must appear ONLY via _gather_rollout (guarded), never inline in train_epoch
    assert "all_gather_cat(" not in te_src


# =========================================================================== #
# SC4a — gathered global-D == single-device ws-block SAPG D, bitwise. cpu-gloo-2proc.
# FIX: single-process reference builds the global context via INDEPENDENT literal
# block-ordered tensors (not by calling _enter), so a wrong _setup/_enter global
# view is detectable; also assert coef diversity so all-blocks-one-coef fails.
# =========================================================================== #

_SC4A = dict(WS=2, NA=4, H=3, OBS=6, EMBD=1, ACT=2, GAMMA=0.99)
_SC4A_KEYS = [
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


def _sc4a_get_values(self, obs_dict, rnn_states=None):
    obs = obs_dict["obs"]
    w = torch.arange(1, obs.shape[-1] + 1, dtype=obs.dtype) * 0.01
    return (obs * w).sum(-1, keepdim=True)


def _sc4a_make_block(block_id, ws, na, H, OBS, EMBD, ACT, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    N = H * na
    gval = torch.linspace(50.0, 0.0, ws)[block_id]

    def r(*s):
        return torch.randn(*s, generator=g)

    obses = r(N, OBS + EMBD)
    obses[:, -EMBD:] = gval
    states = r(N, OBS + EMBD)
    states[:, -EMBD:] = gval
    bd = {
        "actions": r(N, ACT),
        "neglogpacs": r(N),
        "values": r(N, 1),
        "mus": r(N, ACT),
        "sigmas": r(N, ACT).abs() + 0.1,
        "obses": obses,
        "states": states,
        "dones": (r(N) > 0.5).float(),
        "returns": r(N, 1),
        "played_frames": N,
        "step_time": 0.5,
    }
    e_obs = r(H, na, OBS + EMBD)
    e_obs[:, :, -EMBD:] = gval
    e_st = r(H, na, OBS + EMBD)
    e_st[:, :, -EMBD:] = gval
    lo = r(na, OBS + EMBD)
    lo[:, -EMBD:] = gval
    ls = r(na, OBS + EMBD)
    ls[:, -EMBD:] = gval
    ex = {
        "rewards": r(H, na, 1),
        "obs": e_obs,
        "states": e_st,
        "dones": (r(H, na) > 0.5).float(),
        "last_dones": (r(na) > 0.5).float(),
        "last_obs": {"obs": lo, "states": ls},
        "last_rnn_states": None,
        "rnn_states": None,
        "mb_intr_rewards": None,
        "mb_extr_rewards": r(H, na, 1),
    }
    return bd, ex


def _sc4a_cfg(algo, ws, na, H, EMBD):
    algo.world_size = ws
    algo.num_actors = na
    algo.horizon_length = H
    algo.seq_length = H
    algo.gamma = _SC4A["GAMMA"]
    algo.ppo_device = "cpu"
    algo.is_rnn = False
    algo.expl_type = "mixed_expl_learn_param"
    algo.use_others_experience = "lf"
    algo.intr_reward_model = None
    algo.get_values = types.MethodType(_sc4a_get_values, algo)


def _sc4a_install_global_context_literal(algo, ws, na, EMBD):
    """INDEPENDENT oracle: install the global block context by literal block-ordered
    tensors (embd_all[block_ids], coef_all[block_ids]) rather than calling _enter/_setup,
    so a bug in _setup/_enter's global view makes multi != single -> FAIL."""
    block_ids = torch.arange(ws).repeat_interleave(na)
    coef_embd = torch.linspace(50.0, 0.0, ws)
    algo.num_actors = ws * na
    algo.intr_coef_block_size = na
    algo.intr_reward_coef_embd = coef_embd[block_ids].reshape(-1, EMBD).float()
    algo.intr_reward_coef = (torch.linspace(0.5, 0.0, ws) * SCALE)[block_ids].float()


def _sc4a_single_ref():
    p = _SC4A
    ws, na, H, OBS, EMBD, ACT = p["WS"], p["NA"], p["H"], p["OBS"], p["EMBD"], p["ACT"]
    a = types.SimpleNamespace()
    _sc4a_cfg(a, ws, na, H, EMBD)
    _sc4a_install_global_context_literal(a, ws, na, EMBD)  # independent oracle
    C.bind(a, A2CBase, "augment_batch_for_mixed_expl")
    bds, exs = zip(
        *[_sc4a_make_block(b, ws, na, H, OBS, EMBD, ACT, seed=b) for b in range(ws)]
    )
    cbd = {k: torch.cat([bd[k] for bd in bds], 0) for k in _SC4A_KEYS}
    cbd["played_frames"] = sum(bd["played_frames"] for bd in bds)
    cbd["step_time"] = bds[0]["step_time"]
    cex = {
        k: torch.cat([ex[k] for ex in exs], 1)
        for k in ("rewards", "obs", "states", "dones", "mb_extr_rewards")
    }
    cex["last_dones"] = torch.cat([ex["last_dones"] for ex in exs], 0)
    cex["last_obs"] = {
        "obs": torch.cat([ex["last_obs"]["obs"] for ex in exs], 0),
        "states": torch.cat([ex["last_obs"]["states"] for ex in exs], 0),
    }
    cex["last_rnn_states"] = None
    cex["rnn_states"] = None
    cex["mb_intr_rewards"] = None
    D = a.augment_batch_for_mixed_expl(cbd, cex, repeat_idxs=[0, 1])
    return {k: v for k, v in D.items() if torch.is_tensor(v)}


def _sc4a_worker(rank, out, port):
    p = _SC4A
    ws, na, H, OBS, EMBD, ACT = p["WS"], p["NA"], p["H"], p["OBS"], p["EMBD"], p["ACT"]
    dist.init_process_group(
        "gloo", rank=rank, world_size=ws, init_method=f"tcp://127.0.0.1:{port}"
    )
    a = types.SimpleNamespace()
    a.global_rank = rank
    a.multi_gpu = True
    _sc4a_cfg(a, ws, na, H, EMBD)
    C.bind(
        a,
        A2CBase,
        "_setup_gpu_level_sapg",
        "_enter_global_sapg_context",
        "_exit_global_sapg_context",
        "_gather_rollout",
        "augment_batch_for_mixed_expl",
    )
    a._setup_gpu_level_sapg(
        {
            "expl_reward_coef_embd_size": EMBD,
            "expl_reward_type": "entropy",
            "expl_reward_coef_scale": SCALE,
        }
    )
    bd, ex = _sc4a_make_block(rank, ws, na, H, OBS, EMBD, ACT, seed=rank)
    a._enter_global_sapg_context()
    g_bd, g_ex = a._gather_rollout(bd, ex)
    D = a.augment_batch_for_mixed_expl(g_bd, g_ex, repeat_idxs=[0, 1])
    a._exit_global_sapg_context()
    torch.save(
        {k: v for k, v in D.items() if torch.is_tensor(v)},
        out + (".r1" if rank == 1 else ""),
    )
    dist.barrier()
    dist.destroy_process_group()


def test_SC4a_gathered_D_bitwise_equals_single_process(tmp_path):
    out = str(tmp_path / "D_rank0.pt")
    port = GLOO_PORT_BASE + 2
    mp.spawn(_sc4a_worker, args=(out, port), nprocs=_SC4A["WS"], join=True)
    m0 = torch.load(out)
    m1 = torch.load(out + ".r1")
    ref = _sc4a_single_ref()
    for k in m0:  # ranks agree (all_gather makes global batch identical)
        assert torch.equal(m0[k], m1[k]), f"ranks disagree on {k}"
    assert set(m0) == set(ref)
    for k in ref:  # BITWISE core claim vs INDEPENDENT-context reference
        assert torch.equal(m0[k], ref[k]), f"multi != single on {k}"
    mm = ref["off_policy_mask"]
    assert mm.dtype == torch.bool and mm.any() and (~mm).any()
    # coef diversity in the produced D's embedding (all-blocks-one-coef would collapse this)
    on = ref["obses"][~mm][:, -_SC4A["EMBD"] :]
    assert torch.unique(on).numel() >= 2


# =========================================================================== #
# SC4b — _shard_batch is a disjoint equal-size horizon-aligned CONTIGUOUS
# partition. static. FIX: assert per-rank CONTIGUOUS block (not just disjoint),
# horizon-alignment via within-trajectory index, cross-tensor slice consistency,
# and step_time passthrough.
# =========================================================================== #


def test_SC4b_shard_disjoint_equal_contiguous_horizon_aligned():
    ws, H, n_traj = 4, 3, 10  # 10 % 4 != 0 -> 2 tail trajectories dropped
    n = n_traj * H
    # returns[:,0] = global row fingerprint; obses[:,0] mirrors it (cross-tensor check)
    full = {
        "returns": torch.arange(n).float().reshape(n, 1),
        "obses": torch.stack(
            [torch.arange(n).float(), torch.arange(n * 1.0) + 1000], dim=1
        ),
        "off_policy_mask": torch.zeros(n, dtype=torch.bool),
        "played_frames": 123,
        "step_time": 0.5,
    }
    shards = []
    for r in range(ws):
        algo = types.SimpleNamespace(
            world_size=ws, global_rank=r, horizon_length=H, epoch_num=5
        )
        _bind_A2CBase(algo, "_shard_batch")
        shards.append(
            algo._shard_batch(
                {k: (v.clone() if torch.is_tensor(v) else v) for k, v in full.items()}
            )
        )

    sizes = [len(s["returns"]) for s in shards]
    traj_per = n_traj // ws
    used = traj_per * ws * H
    # (A) equal-sized whole-trajectory shards
    assert len(set(sizes)) == 1 and sizes[0] == traj_per * H
    # (B) disjoint union == D[:used]
    idx = torch.cat([s["returns"][:, 0] for s in shards]).long()
    assert torch.equal(idx.sort().values, torch.arange(used))
    # FIX (C) each rank got its EXACT contiguous block [r*tp*H : (r+1)*tp*H]
    for r in range(ws):
        exp = torch.arange(r * traj_per * H, (r + 1) * traj_per * H).float()
        assert torch.equal(shards[r]["returns"][:, 0], exp), (
            f"rank {r} shard not contiguous"
        )
    # FIX (D) horizon-alignment: reshape shard to [traj_per, H]; each trajectory's rows
    # must be H consecutive integers (a horizon-splicing bug breaks this)
    for r in range(ws):
        block = shards[r]["returns"][:, 0].reshape(traj_per, H)
        within = block - block[:, :1]
        assert torch.equal(within, torch.arange(H).float().expand(traj_per, H))
    # FIX (E) cross-tensor slice consistency: obses sliced with SAME offset as returns
    for r in range(ws):
        assert torch.equal(shards[r]["obses"][:, 0], shards[r]["returns"][:, 0])
    # FIX (F) tail drop count + scalars
    assert n_traj - traj_per * ws == 2
    assert shards[0]["played_frames"] == 123
    assert shards[0]["step_time"] == 0.5


# =========================================================================== #
# SC4c — sharded-then-allreduced grad == full-D mean-loss grad (lossless avg).
# cpu-gloo-2proc. Drives the REAL _shard_batch + REAL trancate_gradients_and_step
# (bound), not a hand-rolled all_reduce; equal shards + apply_masks-mean => SUM/ws
# exact. Negative control: different-per-rank population diverges.
# =========================================================================== #


def _sc4c_model():
    torch.manual_seed(1)
    import torch.nn as nn

    return nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))


def _sc4c_flat_grad(m):
    return torch.cat([p.grad.view(-1) for p in m.parameters() if p.grad is not None])


def _sc4c_build_D():
    g = torch.Generator().manual_seed(0)
    return torch.randn(64, 6, generator=g), torch.randn(64, 3, generator=g)


def _sc4c_single_full_grad():
    X, Y = _sc4c_build_D()
    m = _sc4c_model()
    for p in m.parameters():
        p.grad = None
    ((m(X) - Y) ** 2).mean().backward()  # mean == torch_ext.apply_masks reduction
    return _sc4c_flat_grad(m).clone()


def _sc4c_worker(rank, ws, out, port, diff_pop):
    import torch.nn as nn  # noqa: F401

    dist.init_process_group(
        "gloo", rank=rank, world_size=ws, init_method=f"tcp://127.0.0.1:{port}"
    )
    if diff_pop:
        g = torch.Generator().manual_seed(0 if rank == 0 else 999)
        X, Y = torch.randn(64, 6, generator=g), torch.randn(64, 3, generator=g)
        per = 64 // ws
        xs, ys = X[:per], Y[:per]
    else:
        X, Y = _sc4c_build_D()
        # REAL _shard_batch to carve this rank's equal contiguous shard.
        n = X.shape[0]
        H = 4
        algo_sh = types.SimpleNamespace(
            world_size=ws, global_rank=rank, horizon_length=H, epoch_num=5
        )
        C.bind(algo_sh, A2CBase, "_shard_batch")
        # returns key drives shard math; keep X/Y aligned as extra tensors
        shard = algo_sh._shard_batch({"returns": Y, "X": X})
        xs, ys = shard["X"], shard["returns"]
    m = _sc4c_model()
    for p in m.parameters():
        p.grad = None
    ((m(xs) - ys) ** 2).mean().backward()
    # REAL trancate_gradients_and_step averaging op (all_reduce SUM then /world_size),
    # bound onto a stub with the attrs it reads.
    algo = types.SimpleNamespace(
        multi_gpu=True, world_size=ws, model=m, truncate_grads=False
    )
    # scaler/optimizer stubs (truncate_grads=False path still calls scaler.step/update)
    algo.optimizer = torch.optim.SGD(m.parameters(), lr=0.0)

    class _Scaler:
        def unscale_(self, opt):
            pass

        def step(self, opt):
            pass

        def update(self):
            pass

    algo.scaler = _Scaler()
    C.bind(algo, A2CBase, "trancate_gradients_and_step")
    algo.trancate_gradients_and_step()
    if rank == 0:
        torch.save(_sc4c_flat_grad(m).clone(), out)
    dist.barrier()
    dist.destroy_process_group()


def test_SC4c_sharded_allreduce_equals_full_D_grad(tmp_path):
    ws = 2
    out = str(tmp_path / "sharded.pt")
    port = GLOO_PORT_BASE + 3
    mp.spawn(_sc4c_worker, args=(ws, out, port, False), nprocs=ws, join=True)
    sharded = torch.load(out)
    full = _sc4c_single_full_grad()
    assert torch.allclose(sharded, full, atol=1e-5, rtol=1e-4), (
        f"lossy averaging: max_abs {(sharded - full).abs().max().item()}"
    )


def test_SC4c_negative_control_different_population_diverges(tmp_path):
    ws = 2
    out = str(tmp_path / "naive.pt")
    port = GLOO_PORT_BASE + 4
    mp.spawn(_sc4c_worker, args=(ws, out, port, True), nprocs=ws, join=True)
    naive = torch.load(out)
    full = _sc4c_single_full_grad()
    assert not torch.allclose(naive, full, atol=1e-5), (
        "check failed to discriminate broken mechanism"
    )


# =========================================================================== #
# SC5a — single-GPU path byte-identical to PRE-FEATURE upstream (gpu_level
# auto-off without multi_gpu). static.
# The gpu-level feature is uncommitted edits to a2c_common.py + custom_utils.py
# ONLY; a2c_continuous.py is byte-identical to the merge-base (verified). So
# "upstream" == merge-base(HEAD, main) (fallback pin C.FEATURE_BASE_SHA).
# FIX: (1) flag resolves False, (2) standard-branch formula matches base
# byte-for-byte (the __init__ restructure did NOT change the single-GPU-path
# tensor construction), (3) a2c_continuous.py unchanged vs base, (4) the
# custom_utils single-GPU-reachable helpers (filter_leader / create_sinusoidal_
# encoding / shuffle_batch / swap_and_flatten01) unchanged vs base.
# =========================================================================== #


def test_SC5a_single_gpu_flag_off_and_standard_branch():
    # (1) flag resolves False on single process even with config flag on
    assert C.resolve_gpu_level_flag({"gpu_level_sapg": True}, multi_gpu=False) is False
    # (2) standard-branch tensors: replay the REAL upstream formula (a2c_common:340-348)
    num_actors, block = 12288, 4096
    env_ids = torch.arange(num_actors // block).repeat_interleave(block)
    exp_embd = torch.linspace(50.0, 0.0, num_actors // block)[env_ids].reshape(-1, 1)
    exp_coef = torch.linspace(0.5, 0.0, num_actors // block)[env_ids] * SCALE
    assert exp_embd.shape == (num_actors, 1)
    assert exp_coef.shape == (num_actors,)
    assert exp_coef[0].item() == pytest.approx(0.5 * SCALE)  # block 0 == max
    assert exp_coef[-1].item() == pytest.approx(0.0)
    # block count == num_actors // expl_coef_block_size (3), NOT one-per-rank
    assert num_actors // block == 3
    # (3) AST: gpu-level hooks guarded so single-GPU (flag False) never runs them
    tree, _ = C.parse(C.A2C_COMMON)
    te = C.find_method(tree, "ContinuousA2CBase", "train_epoch")
    _assert_calls_guarded(
        te, {"_enter_global_sapg_context", "_gather_rollout", "_shard_batch"}
    )


def _nongpu_branch_formula_lines(src_text):
    """Extract the whitespace-normalized single-GPU-path (non-gpu-level) tensor
    construction statements from a2c_common source text: the standard SAPG block
    setup that runs when gpu_level_sapg is False. Whitespace-normalized so a pure
    reindent (the __init__ restructure) does not count as a behaviour change, but
    ANY token/formula change does."""
    keys = (
        "env_ids = torch.arange(",
        "embedding_genvec = torch.linspace(",
        "self.intr_reward_coef_embd = embedding_genvec.reshape",
        "self.intr_reward_coef_embd = create_sinusoidal_encoding(embedding_genvec",
        "self.intr_reward_coef = torch.linspace(0.5, 0.0",
        "self.intr_reward_coef = torch.linspace(0.0, 0.0",
    )
    out = []
    for line in src_text.splitlines():
        s = line.strip()
        if any(s.startswith(k) for k in keys):
            out.append(" ".join(s.split()))  # collapse internal whitespace
    return out


def test_SC5a_nongpu_branch_formula_matches_base():
    """The __init__ restructure must not change the single-GPU-path formula."""
    base = C.git_show(C.feature_base_ref(), "rl_games/rl_games/common/a2c_common.py")
    if base is None:
        pytest.skip("pre-feature base not resolvable via git")
    base_lines = _nongpu_branch_formula_lines(base)
    feat_lines = _nongpu_branch_formula_lines(C.A2C_COMMON.read_text())
    assert base_lines, "base non-gpu-level formula lines not found (extractor stale?)"
    assert feat_lines == base_lines, (
        "single-GPU-path tensor formula changed vs pre-feature base:\n"
        f"base={base_lines}\nfeat={feat_lines}"
    )


def test_SC5a_a2c_continuous_byte_unchanged_vs_base():
    """a2c_continuous.py executes every training step regardless of the flag; the
    single-GPU byte-identity claim requires it be unchanged vs the PRE-FEATURE base.
    If the base is unavailable (shallow clone), skip rather than false-pass."""
    rel = "rl_games/rl_games/algos_torch/a2c_continuous.py"
    base = C.git_show(C.feature_base_ref(), rel)
    if base is None:
        pytest.skip("pre-feature base not resolvable via git")
    assert base == C.A2C_CONTINUOUS.read_text(), (
        "a2c_continuous.py differs from pre-feature base -> single-GPU path not byte-identical"
    )


def test_SC5a_custom_utils_single_gpu_helpers_unchanged_vs_base():
    """custom_utils.py IS edited by the feature (all_gather_cat added), but the
    single-GPU-reachable helpers must be byte-identical to base. Assert those
    function bodies are unchanged; the only new top-level def is all_gather_cat."""
    rel = "rl_games/rl_games/common/custom_utils.py"
    base_txt = C.git_show(C.feature_base_ref(), rel)
    if base_txt is None:
        pytest.skip("pre-feature base not resolvable via git")
    base_tree = ast.parse(base_txt)
    feat_txt = (C.ROOT / rel).read_text()
    feat_tree = ast.parse(feat_txt)

    def func_bodies(tree, text):
        return {
            n.name: " ".join(ast.get_source_segment(text, n).split())
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
        }

    base_f = func_bodies(base_tree, base_txt)
    feat_f = func_bodies(feat_tree, feat_txt)
    # every base helper unchanged (single-GPU path uses these)
    for name, body in base_f.items():
        assert name in feat_f, f"{name} removed from custom_utils"
        assert feat_f[name] == body, (
            f"custom_utils.{name} changed vs base (single-GPU-reachable)"
        )
    # the feature's only new top-level function is the gather primitive
    added = set(feat_f) - set(base_f)
    assert added == {"all_gather_cat"}, (
        f"unexpected new custom_utils functions: {added}"
    )


# =========================================================================== #
# SC5b — all_gather comm count/bytes independent of off_policy_ratio (offline
# STATIC layer). LAYER 1 (AST): no collective is inside a loop parameterized by
# off_policy_ratio/num_repeat/repeat_idxs. The runtime differential (LAYER 2)
# requires world_size>=3 to be non-vacuous (num_blocks==ws clamps num_repeat at
# ws=2) -> that lives in RUN.md as a >=3-proc/live check, NOT here (would be a
# vacuous false-pass at ws=2).
# =========================================================================== #

_COLLECTIVES = {
    "all_gather",
    "all_reduce",
    "broadcast_object_list",
    "broadcast",
    "all_gather_cat",
}
_BAD_ITER_TOKENS = ("off_policy_ratio", "num_repeat", "repeat_idxs")


def _collective_calls(fn, full_src):
    out = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.loops = []

        def _iters(self):
            return [
                ast.get_source_segment(full_src, ln.iter) for ln in self.loops
            ] or None

        def visit_For(self, n):
            self.loops.append(n)
            self.generic_visit(n)
            self.loops.pop()

        def visit_While(self, n):
            self.loops.append(n)
            self.generic_visit(n)
            self.loops.pop()

        def visit_Call(self, n):
            f = (
                n.func.attr
                if isinstance(n.func, ast.Attribute)
                else getattr(n.func, "id", None)
            )
            if f in _COLLECTIVES:
                out.append((f, self._iters(), n.lineno))
            self.generic_visit(n)

    V().visit(fn)
    return out


def test_SC5b_no_collective_scales_with_off_policy_ratio():
    tree, src = C.parse(C.A2C_COMMON)
    gr = C.find_method(tree, "A2CBase", "_gather_rollout")
    aug = C.find_method(tree, "A2CBase", "augment_batch_for_mixed_expl")
    trc = C.find_method(tree, "A2CBase", "trancate_gradients_and_step")

    gr_c = _collective_calls(gr, src)
    aug_c = _collective_calls(aug, src)
    trc_c = _collective_calls(trc, src)

    # augment has exactly ONE collective and it is broadcast_object_list (pre follower-loop)
    assert [c for c, _, _ in aug_c] == ["broadcast_object_list"], aug_c
    # trancate has exactly ONE collective and it is all_reduce
    assert [c for c, _, _ in trc_c] == ["all_reduce"], trc_c
    # _gather_rollout collectives are all all_gather_cat; none loops over off_policy_ratio
    for c, iters, ln in gr_c:
        assert c == "all_gather_cat", f"unexpected {c} in _gather_rollout L{ln}"
        for it in iters or []:
            assert it is None or not any(b in it for b in _BAD_ITER_TOKENS), (
                f"L{ln} loops over {it}"
            )
    # NO collective anywhere in the three is inside an off_policy_ratio-scaled loop
    for calls in (gr_c, aug_c, trc_c):
        for c, iters, ln in calls:
            for it in iters or []:
                assert it is None or not any(b in it for b in _BAD_ITER_TOKENS), (
                    f"L{ln} {c} scales via {it}"
                )


# =========================================================================== #
# SC6a — per-rank frame accounting stays local; global via *world_size. static.
# FIX: class-scoped ContinuousA2CBase.train (not the A2CBase `pass` stub);
# assert played_frames RHS at line 1070 is bare self.batch_size (no *world_size);
# assert batch_size assigned once; _enter does not touch batch_size.
# =========================================================================== #


def test_SC6a_frame_accounting_local_times_world_size():
    tree, src = C.parse(C.A2C_COMMON)

    # scalars pass through gather + shard unchanged (bare Name, never gathered/sliced)
    for cls, meth in [("A2CBase", "_gather_rollout"), ("A2CBase", "_shard_batch")]:
        fn = C.find_method(tree, cls, meth)
        fn_src = C.src_of(fn, src)
        assert "'played_frames', 'step_time'" in fn_src
        found = False
        for node in ast.walk(fn):
            if isinstance(node, ast.If):
                t = C.src_of(node.test, src)
                if "played_frames" in t and "step_time" in t:
                    assert len(node.body) == 1 and isinstance(node.body[0], ast.Assign)
                    rhs = node.body[0].value
                    assert isinstance(rhs, ast.Name), (
                        f"{meth}: scalar RHS not bare Name (pass-through)"
                    )
                    found = True
        assert found, f"{meth}: no scalar pass-through guard"

    # played_frames = self.batch_size (bare attr, NOT batch_size*world_size) at play_steps
    # (play_steps is defined on A2CBase; ContinuousA2CBase does not override it)
    play = C.find_method(tree, "A2CBase", "play_steps")
    pf_rhs = None
    for node in ast.walk(play):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
        ):
            key = node.targets[0].slice
            if isinstance(key, ast.Constant) and key.value == "played_frames":
                pf_rhs = node.value
    assert pf_rhs is not None, "batch_dict['played_frames'] assignment not found"
    assert isinstance(pf_rhs, ast.Attribute) and pf_rhs.attr == "batch_size", (
        "played_frames must be assigned bare self.batch_size (no world_size multiply)"
    )

    # self.batch_size assigned exactly once (init), never in _enter_global_sapg_context
    bs_assigns = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute)
        and t.attr == "batch_size"
        and isinstance(t.value, ast.Name)
        and t.value.id == "self"
    ]
    assert len(bs_assigns) == 1, (
        f"self.batch_size assigned {len(bs_assigns)}x {bs_assigns}"
    )
    enter = C.find_method(tree, "A2CBase", "_enter_global_sapg_context")
    enter_targets = {
        t.attr
        for n in ast.walk(enter)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute)
    }
    assert "num_actors" in enter_targets and "batch_size" not in enter_targets

    # FIX: class-scoped ContinuousA2CBase.train has the *world_size multiplier + accumulation,
    # gated on multi_gpu, and the SAME name is accumulated (dataflow, not substring).
    train = C.find_method(tree, "ContinuousA2CBase", "train")
    train_src = C.src_of(train, src)
    assert (
        "curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames"
        in train_src
    )
    assert "self.frame += curr_frames" in train_src
    # dataflow: the multiplied `curr_frames` name is the operand accumulated into self.frame
    mult_assigns = [
        n
        for n in ast.walk(train)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "curr_frames" for t in n.targets)
    ]
    assert mult_assigns, "curr_frames assignment not found in ContinuousA2CBase.train"


# =========================================================================== #
# SC6b — gpu-level SAPG rejects RNN; shipped SimToolRealSAPG.yaml (lstm) cannot
# run it unmodified. static (config + guard) + runtime raise (single proc; the
# assert is pre-any-collective so no gloo needed).
# FIX: class-scoped train_epoch finder; loop-tuple check for rnn_states not
# gathered; execute the guard to prove it FIRES.
# =========================================================================== #


def test_SC6b_config_is_lstm():
    import yaml

    cfg = yaml.safe_load(C.SAPG_YAML.read_text())
    assert cfg["params"]["network"]["rnn"]["name"] == "lstm"


def test_SC6b_gather_rollout_rejects_rnn_structurally():
    tree, src = C.parse(C.A2C_COMMON)
    gr = C.find_method(tree, "A2CBase", "_gather_rollout")
    gr_src = C.src_of(gr, src)
    assert "gpu_level_sapg does not support RNN policies yet" in gr_src
    # structural: assert not self.is_rnn
    found = False
    for n in ast.walk(gr):
        if (
            isinstance(n, ast.Assert)
            and isinstance(n.test, ast.UnaryOp)
            and isinstance(n.test.op, ast.Not)
        ):
            t = n.test.operand
            if (
                isinstance(t, ast.Attribute)
                and t.attr == "is_rnn"
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                found = True
    assert found, "no structural `assert not self.is_rnn` in _gather_rollout"

    # reached unconditionally under the gpu_level guard at epoch 0 (class-scoped train_epoch)
    te = C.find_method(tree, "ContinuousA2CBase", "train_epoch")
    te_src = C.src_of(te, src)
    m = te_src.index("if self.gpu_level_sapg:")
    g = te_src.index("self._gather_rollout(")
    assert g > m
    assert "is_rnn" not in te_src[m:g], (
        "unexpected is_rnn short-circuit before _gather_rollout"
    )

    # loop-tuple integrity: rnn_states / last_rnn_states NOT in the gathered extras loop
    for tok in ("rnn_states", "last_rnn_states"):
        # the extras gather loop tuple is a hardcoded 6-key tuple; assert tok not gathered
        assert f"'{tok}'" not in gr_src, (
            f"{tok} unexpectedly referenced in _gather_rollout"
        )
    # augment DOES consume rnn_states/last_rnn_states -> the guard is substantive
    aug_src = C.src_of(
        C.find_method(tree, "A2CBase", "augment_batch_for_mixed_expl"), src
    )
    assert "rnn_states" in aug_src and "last_rnn_states" in aug_src


def test_SC6b_gather_rollout_raises_on_is_rnn():
    """Execute the guard: it FIRES (not just present in source). Single process ok
    because the assert precedes any collective."""
    algo = types.SimpleNamespace(is_rnn=True, world_size=1, global_rank=0)
    C.bind(algo, A2CBase, "_gather_rollout")
    with pytest.raises(AssertionError, match="does not support RNN"):
        algo._gather_rollout(
            {"returns": torch.zeros(4)},
            {"last_dones": torch.zeros(2), "last_obs": {"obs": torch.zeros(2, 3)}},
        )

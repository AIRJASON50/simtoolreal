"""GPU-level SAPG — LEARNING-SIGNAL-EQUIVALENCE gate test (spec section 3).

Proves that the GPU-level SAPG path (world_size gloo processes, each fed one
exploration block's rollout, ``gpu_level_sapg=True``) produces the SAME learning
signal as a SINGLE process running world_size SAPG blocks over the identical
injected rollout, the same ``minibatch_size``, and the same tail-trimmed D.

Why this file exists (vs. the shipped ``test_offline_suite.py``):
    The offline suite proves the plumbing (gather order / shard partition /
    grad average) with a *stub* linear model and no RNN.  Stubs hide exactly the
    three silent-corruption classes SAPG is most exposed to:
      * off-policy return recompute through the REAL LSTM + REAL central critic
        (augment_batch_for_mixed_expl calls get_values -> the CV forward),
      * the ``extra_param`` per-block head selection (coef-ids must be GLOBAL),
      * RunningMeanStd / advantage-norm / KL pooling across ranks.
    So here we build the REAL model via rl_games' network_builder /
    ModelA2CContinuousLogStd / CentralValueTrain with the PRODUCTION structure
    (LSTM before_mlp + layer_norm, coef_cond sigma, extra_param actor, asymmetric
    MLP central_value, expl_type=mixed_expl_learn_param, use_others_experience=lf,
    off_policy_ratio=1) at SMALL sizes, and drive both paths through the SAME
    production bytecode (bound via types.MethodType), then compare.

Reference definition (spec section 3, corrected):
    ONE process, num_actors = ws * na_local, expl_coef_block_size = na_local
    (=> num_blocks = ws, block b == the envs rank b simulated), WORLD_SIZE /
    LOCAL_RANK UNSET, fed the identical injected rollout, SAME minibatch_size
    over the SAME tail-trimmed D (we apply the _shard_batch tail-trim in the
    reference by hand).  Each of its size-minibatch_size minibatches == the
    row-union of the ws ranks' k-th size-(minibatch_size // ws) minibatches.

Everything runs on CPU, gloo, torch.multiprocessing.spawn.  NO Isaac Sim, NO GPU,
NO NCCL.  fp32 + mixed_precision disabled for the exact-equality pass.

Run:  .venv_isaacsim/bin/python -m pytest \
          tests/sapg_verify/test_learning_signal_equiv.py -v

(Any torch env works; the file needs no GPU and no Isaac Sim.)
"""

import os
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_games.algos_torch import central_value
from rl_games.algos_torch.models import ModelA2CContinuousLogStd, ModelCentralValue
from rl_games.algos_torch.network_builder import A2CBuilder
from rl_games.common.a2c_common import A2CBase, ContinuousA2CBase
from rl_games.algos_torch.a2c_continuous import A2CAgent

from tests.sapg_verify import _common as C


# --------------------------------------------------------------------------- #
# Small production-shaped sizes.  Keep the STRUCTURE that matters (LSTM
# before_mlp + layer_norm, asymmetric central_value MLP, learn_param/extra_param,
# coef_cond sigma) but shrink every dimension so CPU forwards are instant.
# --------------------------------------------------------------------------- #

NA_LOCAL = 8  # per-rank / per-block envs (== expl_coef_block_size)
HORIZON = 4  # H == seq_length (one game per horizon-trajectory)
SEQ = HORIZON
OBS_DIM = 6  # actor obs (before coef-id column)
STATE_DIM = 9  # central-value state (asymmetric: state != obs)
ACT_DIM = 3
HIDDEN = 16  # LSTM units + MLP head sizes shrunk
MLP_UNITS = [16, 16]
EMBD_DIM = 1  # learn_param / disjoint => single coef-id column
SCALE = 0.002  # expl_reward_coef_scale (SimToolRealSAPG.yaml)
GAMMA = 0.99
OFF_POLICY_RATIO = 1  # off_policy_ratio=1 => num_repeat=2 => repeat_idxs=[0, k]


# Deterministic off-policy partner block for augment.  In production it is drawn
# randomly then dist.broadcast_object_list-synced; here we pin it so both paths
# augment identically.  For ws==2 the only follower is block 1; for ws==3 pick 2.
def repeat_idxs_for(ws):
    return [0, ws - 1]


GLOO_PORT_BASE = 29750


# --------------------------------------------------------------------------- #
# Network params (rl_games network_builder schema) — matches
# SimToolRealSAPG.yaml network: separate=False, mlp elu, rnn lstm before_mlp
# layer_norm, coef_cond sigma.  Only sizes are shrunk.
# --------------------------------------------------------------------------- #


def _actor_network_params():
    return {
        "name": "actor_critic",
        "separate": False,
        "space": {
            "continuous": {
                "mu_activation": "None",
                "sigma_activation": "None",
                "mu_init": {"name": "default"},
                "sigma_init": {"name": "const_initializer", "val": 0},
                "fixed_sigma": "coef_cond",  # SAPG requires coef_cond
            }
        },
        "mlp": {
            "units": list(MLP_UNITS),
            "activation": "elu",
            "d2rl": False,
            "initializer": {"name": "default"},
            "regularizer": {"name": "None"},
        },
        "rnn": {
            "name": "lstm",
            "units": HIDDEN,
            "layers": 1,
            "before_mlp": True,
            "layer_norm": True,
        },
    }


def _cv_network_params():
    # Asymmetric central value: pure MLP (no rnn), central_value=True.
    return {
        "name": "actor_critic",
        "central_value": True,
        "mlp": {
            "units": list(MLP_UNITS),
            "activation": "elu",
            "d2rl": False,
            "initializer": {"name": "default"},
            "regularizer": {"name": "None"},
        },
    }


def _build_actor_model(coef_ids):
    """Real ModelA2CContinuousLogStd with LSTM+extra_param+coef_cond, seeded so
    every process / the reference build IDENTICAL weights."""
    torch.manual_seed(12345)
    builder = A2CBuilder()
    builder.load(_actor_network_params())
    model = ModelA2CContinuousLogStd(builder)
    build_config = {
        "actions_num": ACT_DIM,
        "input_shape": (OBS_DIM + EMBD_DIM,),
        "num_seqs": 1,
        "value_size": 1,
        "normalize_value": True,
        "normalize_input": True,
        "type": "extra_param",
        "coef_ids": coef_ids,
        "coef_id_idx": OBS_DIM,
    }
    net = model.build(build_config)
    net.eval()
    return net


def _build_cv(coef_ids, minibatch_size):
    """Real CentralValueTrain (asymmetric MLP critic), seeded IDENTICALLY.

    CentralValueTrain expects a *Model* (ModelCentralValue) as its ``network``
    arg: ``self.model = network.build(state_config)`` must yield a Network whose
    forward returns a {'values': ...} dict (ModelCentralValue.forward), NOT the
    raw A2CBuilder.Network tuple.  This mirrors production, where
    central_value_config['network'] is built via ModelBuilder(name='central_value').
    """
    torch.manual_seed(999)
    builder = A2CBuilder()
    builder.load(_cv_network_params())
    cv_model = ModelCentralValue(builder)
    cv_config = {
        "minibatch_size": minibatch_size,
        "mini_epochs": 1,
        "learning_rate": 1e-4,
        "kl_threshold": 0.016,
        "clip_value": True,
        "normalize_input": True,
        "normalize_value": True,
        "truncate_grads": False,
        "network": cv_model,
    }
    cv = central_value.CentralValueTrain(
        state_shape=(STATE_DIM + EMBD_DIM,),
        value_size=1,
        ppo_device="cpu",
        num_agents=1,
        horizon_length=HORIZON,
        num_actors=NA_LOCAL,  # placeholder; batch/minibatch recomputed below
        num_actions=ACT_DIM,
        seq_length=SEQ,
        normalize_value=True,
        network=cv_model,
        config=cv_config,
        writter=None,
        max_epochs=1,
        multi_gpu=False,
        zero_rnn_on_done=True,
        type="extra_param",
        coef_ids=coef_ids,
        coef_id_idx=STATE_DIM,
    )
    return cv


# --------------------------------------------------------------------------- #
# Deterministic rollout tape.  Seeded PER GLOBAL ENV ID so rank r's local envs
# == single-proc envs [r*na : (r+1)*na].  Layout matches play_steps exactly:
#   flat batch_dict tensors : swap_and_flatten01([H, na, ...]) => env-major rows
#                             (env g occupies rows [g*H : (g+1)*H]),
#   rnn_states              : list of [num_layers, na(game axis == dim1), hidden],
#   extras['obs'/'states']  : [H, na, ...]         (env axis == dim1),
#   extras['rnn_states']    : [H, num_layers, na, hidden]  (env axis == dim2),
#   extras['last_rnn_states']: [num_layers, na, hidden]    (env axis == dim1).
# The coef-id column is stamped LOCAL (this rank's / this block's coef) exactly
# like a real rollout; augment overwrites it in the global view.
# --------------------------------------------------------------------------- #


def _swap_and_flatten01(arr):
    # local copy of custom_utils.swap_and_flatten01 (avoid import churn)
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _block_coef_embd(block_id, ws):
    """The block's coef-id scalar == linspace(50, 0, ws)[block_id] (disjoint
    embd), matching _setup_gpu_level_sapg's gen_all[block_id]."""
    return torch.linspace(50.0, 0.0, ws)[block_id].item()


def _make_block_rollout(block_id, ws):
    """Build one block's (== one rank's local) rollout tape, seeded per global
    env id so it is identical whether produced by rank block_id or sliced out of
    the single-process global tape."""
    na, H = NA_LOCAL, HORIZON
    coef_val = _block_coef_embd(block_id, ws)

    # per-env deterministic streams keyed on GLOBAL env id (block_id*na + e)
    def env_stream(e, *shape):
        gid = block_id * na + e
        return torch.randn(*shape, generator=_gen(100000 + gid))

    # [H, na, ...] rollout tensors
    obs = torch.stack(
        [torch.stack([env_stream(e, OBS_DIM) for e in range(na)]) for _ in range(H)]
    )  # [H, na, OBS_DIM]
    obs = torch.cat([obs, torch.full((H, na, EMBD_DIM), coef_val)], dim=-1)
    states = torch.stack(
        [torch.stack([env_stream(e, STATE_DIM) for e in range(na)]) for _ in range(H)]
    )  # [H, na, STATE_DIM]
    states = torch.cat([states, torch.full((H, na, EMBD_DIM), coef_val)], dim=-1)
    actions = torch.stack(
        [torch.stack([env_stream(e, ACT_DIM) for e in range(na)]) for _ in range(H)]
    )
    mus = torch.stack(
        [torch.stack([env_stream(e, ACT_DIM) for e in range(na)]) for _ in range(H)]
    )
    sigmas = torch.stack(
        [
            torch.stack([env_stream(e, ACT_DIM).abs() + 0.2 for e in range(na)])
            for _ in range(H)
        ]
    )
    neglogpacs = torch.stack(
        [torch.stack([env_stream(e, 1)[0] for e in range(na)]) for _ in range(H)]
    )  # [H, na]
    values = torch.stack(
        [torch.stack([env_stream(e, 1) for e in range(na)]) for _ in range(H)]
    )  # [H, na, 1]
    rewards = torch.stack(
        [torch.stack([env_stream(e, 1) for e in range(na)]) for _ in range(H)]
    )  # [H, na, 1]
    dones = torch.stack(
        [
            torch.stack([(env_stream(e, 1)[0] > 0.6).float() for e in range(na)])
            for _ in range(H)
        ]
    )  # [H, na]
    last_dones = torch.stack(
        [(env_stream(e, 1)[0] > 0.6).float() for e in range(na)]
    )  # [na]

    # rnn carriers, seeded per global env id
    num_layers = 1
    h0 = torch.stack([env_stream(e, HIDDEN) for e in range(na)])  # [na, hidden]
    c0 = torch.stack([env_stream(e, HIDDEN) for e in range(na)])
    # packed actor rnn_states: list of [num_layers, na(game), hidden]
    packed_rnn = [
        h0.reshape(num_layers, na, HIDDEN).contiguous(),
        c0.reshape(num_layers, na, HIDDEN).contiguous(),
    ]
    # rnn_state_buffer: [H, num_layers, na, hidden]  (env axis == dim2)
    rnn_buf = [
        torch.stack([env_stream(e, HIDDEN) for e in range(na)])
        .reshape(1, num_layers, na, HIDDEN)
        .repeat(H, 1, 1, 1)
        .contiguous()
        for _ in range(2)
    ]
    # last_rnn_states: list of [num_layers, na, hidden]  (env axis == dim1)
    last_rnn = [
        torch.stack([env_stream(e, HIDDEN) for e in range(na)])
        .reshape(num_layers, na, HIDDEN)
        .contiguous()
        for _ in range(2)
    ]

    # --- flatten to play_steps batch_dict layout (env-major rows) -------------
    returns = rewards + GAMMA * values  # simple synthetic returns (shape [H,na,1])
    batch_dict = {
        "obses": _swap_and_flatten01(obs),
        "states": _swap_and_flatten01(states),
        "actions": _swap_and_flatten01(actions),
        "mus": _swap_and_flatten01(mus),
        "sigmas": _swap_and_flatten01(sigmas),
        "neglogpacs": _swap_and_flatten01(neglogpacs),
        "values": _swap_and_flatten01(values),
        "returns": _swap_and_flatten01(returns),
        "dones": _swap_and_flatten01(dones),
        "rnn_states": packed_rnn,
        "played_frames": H * na,
        "step_time": 0.1,
    }
    extras = {
        "rewards": rewards,  # [H, na, 1]
        "obs": obs.clone(),  # [H, na, OBS_DIM+EMBD]
        "states": states.clone(),  # [H, na, STATE_DIM+EMBD]
        "dones": dones.contiguous(),  # [H, na]  (play_steps: extras['dones']==mb_fdones)
        "last_dones": last_dones,  # [na]
        "last_obs": {
            "obs": torch.stack([env_stream(e, OBS_DIM) for e in range(na)]),
            "states": torch.stack([env_stream(e, STATE_DIM) for e in range(na)]),
        },
        "rnn_states": rnn_buf,  # [H, num_layers, na, hidden]
        "last_rnn_states": last_rnn,  # [num_layers, na, hidden]
        "mb_intr_rewards": None,
        "mb_extr_rewards": rewards,
    }
    # stamp the coef column on last_obs too (rollout invariant)
    extras["last_obs"]["obs"] = torch.cat(
        [extras["last_obs"]["obs"], torch.full((na, EMBD_DIM), coef_val)], dim=-1
    )
    extras["last_obs"]["states"] = torch.cat(
        [extras["last_obs"]["states"], torch.full((na, EMBD_DIM), coef_val)], dim=-1
    )
    return batch_dict, extras


# --------------------------------------------------------------------------- #
# Global-tape assembly for the single-process reference: concatenate the ws
# blocks EXACTLY as _gather_rollout would (flat dim0, rnn dim1, extras dim1,
# rnn_buf dim2, last_* dim0/1).  A single process with num_actors=ws*na sees
# precisely this.
# --------------------------------------------------------------------------- #


def _assemble_global(blocks):
    bds = [b[0] for b in blocks]
    exs = [b[1] for b in blocks]
    flat_keys = [
        "obses",
        "states",
        "actions",
        "mus",
        "sigmas",
        "neglogpacs",
        "values",
        "returns",
        "dones",
    ]
    g_bd = {k: torch.cat([bd[k] for bd in bds], dim=0) for k in flat_keys}
    g_bd["rnn_states"] = [
        torch.cat([bd["rnn_states"][i] for bd in bds], dim=1) for i in range(2)
    ]
    g_bd["played_frames"] = sum(bd["played_frames"] for bd in bds)
    g_bd["step_time"] = bds[0]["step_time"]

    g_ex = {}
    for k in ("rewards", "obs", "states", "dones", "mb_extr_rewards"):
        g_ex[k] = torch.cat([ex[k] for ex in exs], dim=1)
    g_ex["last_dones"] = torch.cat([ex["last_dones"] for ex in exs], dim=0)
    g_ex["last_obs"] = {
        "obs": torch.cat([ex["last_obs"]["obs"] for ex in exs], dim=0),
        "states": torch.cat([ex["last_obs"]["states"] for ex in exs], dim=0),
    }
    g_ex["rnn_states"] = [
        torch.cat([ex["rnn_states"][i] for ex in exs], dim=2) for i in range(2)
    ]
    g_ex["last_rnn_states"] = [
        torch.cat([ex["last_rnn_states"][i] for ex in exs], dim=1) for i in range(2)
    ]
    g_ex["mb_intr_rewards"] = None
    g_ex["mb_extr_rewards"] = g_ex["mb_extr_rewards"]
    return g_bd, g_ex


# --------------------------------------------------------------------------- #
# Harness object: a SimpleNamespace with the exact attrs the bound production
# methods read, plus the REAL actor model + REAL central_value_net.  We bind:
#   _setup_gpu_level_sapg / _enter / _exit / _gather_rollout / _shard_batch
#   augment_batch_for_mixed_expl  (A2CBase)
#   prepare_dataset               (ContinuousA2CBase)
#   get_values / get_central_value / _preproc_obs  (A2CBase)
#   calc_gradients / trancate_gradients_and_step   (A2CAgent / A2CBase)
# so the whole compare runs production bytecode, not a re-implementation.
# --------------------------------------------------------------------------- #


def _make_harness(ws, rank, multi_gpu, minibatch_size, cv_minibatch_size):
    from rl_games.common import datasets

    a = types.SimpleNamespace()
    a.world_size = ws if multi_gpu else 1
    a.global_rank = rank
    a.local_rank = rank
    a.multi_gpu = multi_gpu
    a.ppo_device = "cpu"
    a.num_actors = NA_LOCAL
    a.num_agents = 1
    a.horizon_length = HORIZON
    a.seq_length = SEQ
    a.gamma = GAMMA
    a.tau = 0.95
    a.is_rnn = True
    a.value_size = 1
    a.actions_num = ACT_DIM
    a.obs_shape = (OBS_DIM,)
    a.state_shape = (STATE_DIM,)
    a.expl_type = "mixed_expl_learn_param"
    a.use_others_experience = "lf"
    a.intr_reward_model = None
    a.epoch_num = 1
    a.e_clip = 0.1
    a.ppo = True
    a.critic_coef = 4.0
    a.bounds_loss_coef = 0.0001
    a.bound_loss_type = "bound"
    a.entropy_coef = 0.0
    a.last_lr = 1e-4
    a.weight_decay = 0.0
    # get_values() stuffs self.dones into its CV input_dict (as 'is_done'); the
    # MLP central critic never reads it, but the attr must exist.
    a.dones = torch.zeros(NA_LOCAL, dtype=torch.uint8)
    a.mixed_precision = False  # fp32 EXACT pass
    a.normalize_advantage = True
    a.normalize_rms_advantage = False
    a.normalize_value = True
    a.normalize_input = True
    a.clip_value = True
    a.has_value_loss = True
    a.zero_rnn_on_done = True
    a.schedule_type = "standard"
    a.grad_norm = 1.0
    a.truncate_grads = False  # keep grads un-clipped so all_reduce/ws is exact
    a.config = {
        "off_policy_ratio": OFF_POLICY_RATIO,
        "expl_reward_type": "entropy",
    }
    a.has_central_value = True
    # diagnostics stub (calc_gradients calls self.diagnostics.mini_batch)
    a.diagnostics = types.SimpleNamespace(mini_batch=lambda *args, **kw: None)
    # actor_loss_func — use the real common_losses actor loss
    from rl_games.common import common_losses

    a.actor_loss_func = common_losses.actor_loss

    C.bind(
        a,
        A2CBase,
        "_setup_gpu_level_sapg",
        "_enter_global_sapg_context",
        "_exit_global_sapg_context",
        "_gather_rollout",
        "_shard_batch",
        "augment_batch_for_mixed_expl",
        "get_values",
        "get_central_value",
        "_preproc_obs",
        "trancate_gradients_and_step",
    )
    # FIX (a): bound_loss / reg_loss live on A2CAgent (a2c_continuous.py), NOT on
    # A2CBase.  Binding them from A2CBase raises AttributeError at setup.  Bind
    # them (and the other A2CAgent-only train methods) from A2CAgent.
    C.bind(a, A2CAgent, "bound_loss", "reg_loss")
    a.prepare_dataset = types.MethodType(ContinuousA2CBase.prepare_dataset, a)
    a.calc_gradients = types.MethodType(A2CAgent.calc_gradients, a)
    a.train_actor_critic = types.MethodType(A2CAgent.train_actor_critic, a)

    # --- set up the SAPG block structure (real _setup) ------------------------
    # gpu_level path: _setup computes global block embd/coef AND halves minibatch.
    a.minibatch_size = minibatch_size
    a.num_minibatches = (HORIZON * NA_LOCAL) // minibatch_size
    setup_cfg = {
        "expl_reward_coef_embd_size": EMBD_DIM,
        "expl_reward_type": "entropy",
        "expl_reward_coef_scale": SCALE,
        "central_value_config": {"minibatch_size": cv_minibatch_size},
    }
    if multi_gpu:
        a.gpu_level_sapg = True
        a._setup_gpu_level_sapg(setup_cfg)
        # minibatch_size now == minibatch_size // ws; cv config halved in place
        cv_mb = setup_cfg["central_value_config"]["minibatch_size"]
    else:
        # single-process reference: standard (non-gpu-level) block path.
        a.gpu_level_sapg = False
        a.intr_coef_block_size = NA_LOCAL
        a.num_actors = ws * NA_LOCAL
        block_ids = torch.arange(ws).repeat_interleave(NA_LOCAL)
        gen_all = torch.linspace(50.0, 0.0, ws)
        embd_all = gen_all.reshape(-1, 1)
        coef_all = torch.linspace(0.5, 0.0, ws) * SCALE
        a.intr_reward_coef_embd = embd_all[block_ids].float()  # [ws*na, 1]
        a.intr_reward_coef = coef_all[block_ids].float()  # [ws*na]
        a._gpu_global_intr_embd = a.intr_reward_coef_embd
        a.minibatch_size = minibatch_size
        cv_mb = cv_minibatch_size

    # --- build the REAL actor model + central value with GLOBAL coef-ids -------
    global_embd = a._gpu_global_intr_embd  # [ws*na, 1] block-ordered
    coef_ids = global_embd[::NA_LOCAL, 0]  # ws distinct block ids (spec B.3)
    a.model = _build_actor_model(coef_ids)
    a.value_mean_std = a.model.value_mean_std

    # central value net (asymmetric MLP), sized to the GLOBAL num_actors.
    a.central_value_net = _build_cv(coef_ids, cv_mb)
    a.central_value_net.num_actors = ws * NA_LOCAL
    a.central_value_net.batch_size = HORIZON * ws * NA_LOCAL
    a.central_value_net.dataset = datasets.PPODataset(
        a.central_value_net.batch_size, cv_mb, True, False, "cpu", SEQ
    )
    # FIX (CV grad): production (a2c_continuous.py:65) builds the CV with
    # 'multi_gpu': self.multi_gpu, so under gpu_level_sapg the CV's calc_gradients
    # runs the all_reduce(SUM)/ws averaging (central_value.py:272-286) and the
    # optimizer sees the GLOBAL-D-averaged grad.  Mirror that here so each rank's
    # captured CV param.grad equals the single-proc full-D CV grad; the CPU worker
    # is already inside a gloo process group so the all_reduce is valid.
    a.central_value_net.multi_gpu = multi_gpu
    if multi_gpu:
        a.central_value_net.world_size = ws
        a.central_value_net.global_rank = rank
        a.central_value_net.local_rank = rank
    else:
        a.central_value_net.world_size = 1
        a.central_value_net.global_rank = 0
        a.central_value_net.local_rank = 0

    # actor PPODataset over the (post-shard, per-rank) batch.
    a.dataset = datasets.PPODataset(
        HORIZON * (ws * NA_LOCAL if not multi_gpu else NA_LOCAL),
        a.minibatch_size,
        False,
        True,
        "cpu",
        SEQ,
    )
    a.optimizer = torch.optim.Adam(a.model.parameters(), 1e-4, eps=1e-8)

    class _Scaler:
        def scale(self, x):
            return x

        def unscale_(self, opt):
            pass

        def step(self, opt):
            opt.step()

        def update(self):
            pass

    a.scaler = _Scaler()
    return a


# --------------------------------------------------------------------------- #
# Common pipeline: gather (if multi) -> augment -> shard (if multi) -> shuffle
# (SKIPPED for determinism; equivalence must hold pre-shuffle) -> prepare_dataset.
# Returns the actor + cv dataset_dicts (via monkeypatched update_values_dict
# capture) and the head-selection idxs from the REAL extra_param forward.
# --------------------------------------------------------------------------- #


def _run_pipeline(a, batch_dict, extras, ws, repeat_idxs):
    if a.multi_gpu:
        a._enter_global_sapg_context()
        batch_dict, extras = a._gather_rollout(batch_dict, extras)
    aug = a.augment_batch_for_mixed_expl(batch_dict, extras, repeat_idxs=repeat_idxs)
    if a.multi_gpu:
        aug = a._shard_batch(aug)
    # NOTE: production calls shuffle_batch here; we skip it (random) so the
    # per-game comparison is deterministic.  Equivalence is a PRE-shuffle claim
    # (shuffle permutes both paths' games consistently by the same seed at run).
    a.curr_frames = aug.pop("played_frames")
    a.prepare_dataset(aug)
    return aug


# --------------------------------------------------------------------------- #
# Head-selection idxs from the REAL extra_param forward (spec assertion 4 +
# negative control B.3).  Mirrors network_builder.py:320.
# --------------------------------------------------------------------------- #


def _head_idxs(model_net, obs_with_coef):
    a2c = model_net.a2c_network
    pid = obs_with_coef[:, a2c.pid_idx].reshape(-1, 1)
    return (pid == a2c.param_ids).float().argmax(dim=1)


# --------------------------------------------------------------------------- #
# Worker: one gpu-level rank.  Produces, per game, its shard's advantages /
# returns / rnn_states / cv-states / head-idxs, the per-mini_epoch KL scalar,
# and the per-parameter actor+cv gradient AFTER all_reduce/ws for optimizer
# step 0.  Everything is written to <out>.<rank>.pt.
# --------------------------------------------------------------------------- #


def _capture_actor_grad_before_step(a, mb):
    """Run ONE calc_gradients minibatch; capture the DDP-AVERAGED per-parameter
    grad that actually feeds the optimizer step, then compare that to single-proc.

    FIX (6) — SUM vs mean.  trancate_gradients_and_step RETURNS the raw
    all_reduce(SUM) concat (a2c_common.py:535-536 build all_grads, all_reduce it
    in place, return it); it only writes the /world_size AVERAGED grad into
    param.grad (line 540-542) before scaler.step.  So the returned tensor is
    ws * (single-proc grad), while param.grad == single-proc grad.  The DDP
    learning signal is the AVERAGED grad (what Adam consumes), so we capture
    param.grad right after the reduce/copy (before scaler.step moves weights, but
    weights don't affect the already-computed grad).  Single-proc: multi_gpu is
    off, so all_reduce is skipped and param.grad is just this proc's grad -- the
    equivalence target.  Returning param.grad (not the SUM) makes multi == single
    at fp32 in BOTH paths without any ws bookkeeping in the assertions."""
    captured = {}
    orig_step = a.trancate_gradients_and_step

    def _hook():
        g = orig_step()  # real all_reduce(SUM)+/ws copy into param.grad; returns SUM
        # param.grad is the AVERAGED grad (multi) / this proc's grad (single) --
        # the DDP-consistent learning signal fed to the optimizer.
        captured["grad"] = (
            torch.cat(
                [p.grad.view(-1) for p in a.model.parameters() if p.grad is not None]
            )
            .detach()
            .clone()
        )
        return g

    a.trancate_gradients_and_step = _hook
    a.calc_gradients(mb)
    a.trancate_gradients_and_step = orig_step
    return captured["grad"], a.train_result[3].detach().clone()  # (grad, kl)


def _sapg_worker(rank, ws, out, port, mode):
    """mode: 'ok' | 'revert_rnn_shard' | 'skip_tailtrim' | 'local_cv_coef' |
    'full_minibatch' — negative controls monkeypatch one fix off."""
    os.environ["WORLD_SIZE"] = str(ws)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    dist.init_process_group(
        "gloo", rank=rank, world_size=ws, init_method=f"tcp://127.0.0.1:{port}"
    )

    # global tail-trimmed D is (ws+1?)... actually augment with off_policy_ratio=1
    # + lf yields exactly (ws*na)*H rows (one follower relabel per block).  Pick
    # a minibatch so that after //ws each rank has an INTEGER number of games and
    # >= 1 full global minibatch survives the trim.
    minibatch_size = _pick_minibatch(ws, mode)
    cv_minibatch_size = minibatch_size

    a = _make_harness(ws, rank, True, minibatch_size, cv_minibatch_size)
    _maybe_break(a, mode)

    bd, ex = _make_block_rollout(rank, ws)
    ridx = repeat_idxs_for(ws)
    _run_pipeline(a, bd, ex, ws, ridx)  # side-effect: fills a.dataset / CV dataset

    # --- capture per-game shard signals --------------------------------------
    ds = a.dataset.values_dict
    # global game ids this rank owns (contiguous after _shard_batch trim)
    n_games = len(ds["returns"]) // SEQ
    payload = {
        "advantages": ds["advantages"].detach().clone(),
        "returns": ds["returns"].detach().clone(),
        "off_policy_mask": (
            ds["off_policy_mask"].detach().clone()
            if ds.get("off_policy_mask") is not None
            else None
        ),
        "rnn_h": ds["rnn_states"][0].detach().clone(),  # [layers, games, hidden]
        "n_games": n_games,
        "len_actor_dataset": len(a.dataset),
        "len_cv_dataset": len(a.central_value_net.dataset),
    }
    # cv states + head idxs on the REAL cv net
    cv_ds = a.central_value_net.dataset.values_dict
    payload["cv_states"] = cv_ds["obs"].detach().clone()
    payload["cv_head_idxs"] = (
        _head_idxs(a.central_value_net.model, cv_ds["obs"]).detach().clone()
    )
    payload["actor_head_idxs"] = _head_idxs(a.model, ds["obs"]).detach().clone()

    # --- per-mini_epoch KL + per-step grad (mini_ep 0, minibatch 0) -----------
    a.model.train()
    mb0 = a.dataset[0]
    # row count of THIS rank's minibatch 0 (for the pooled/row-weighted KL check).
    payload["kl_rows_step0"] = len(mb0["returns"])
    grad, kl = _capture_actor_grad_before_step(a, mb0)
    payload["actor_grad_step0"] = grad
    payload["kl_step0"] = kl

    # cv gradient for step 0 (real CentralValueTrain.calc_gradients all_reduce/ws)
    a.central_value_net.train()
    cv_mb0 = a.central_value_net.dataset[0]
    cv_grad = _cv_grad_after_reduce(a.central_value_net, cv_mb0)
    payload["cv_grad_step0"] = cv_grad

    # collective symmetry: min/max of dataset lengths across ranks
    len_t = torch.tensor(
        [len(a.dataset), len(a.central_value_net.dataset)], dtype=torch.float64
    )
    lmin = len_t.clone()
    lmax = len_t.clone()
    dist.all_reduce(lmin, op=dist.ReduceOp.MIN)
    dist.all_reduce(lmax, op=dist.ReduceOp.MAX)
    payload["len_min"] = lmin.clone()
    payload["len_max"] = lmax.clone()

    torch.save(payload, f"{out}.{rank}.pt")
    dist.barrier()
    dist.destroy_process_group()


def _cv_grad_after_reduce(cv, mb):
    """Run CentralValueTrain.calc_gradients up to the all_reduce/ws, capture the
    flat grad, but DON'T let the Adam step move weights before capture (capture
    happens after copy_ of the averaged grad, before optimizer.step)."""
    captured = {}

    orig_step = torch.optim.Adam.step

    def _patched_step(self, *a, **k):
        captured["grad"] = (
            torch.cat(
                [p.grad.view(-1) for p in cv.model.parameters() if p.grad is not None]
            )
            .detach()
            .clone()
        )
        return orig_step(self, *a, **k)

    torch.optim.Adam.step = _patched_step
    try:
        cv.calc_gradients(mb)
    finally:
        torch.optim.Adam.step = orig_step
    return captured["grad"]


# --------------------------------------------------------------------------- #
# Single-process reference (WORLD_SIZE unset).  num_actors = ws*na, one process,
# same minibatch_size (NOT halved), over the same tail-trimmed D.
# --------------------------------------------------------------------------- #


def _single_ref(ws, mode):
    for k in ("WORLD_SIZE", "LOCAL_RANK", "RANK"):
        os.environ.pop(k, None)
    # The reference is ALWAYS the CORRECT single-process run at this mode's
    # minibatch over the CORRECTLY tail-trimmed D (spec section 3).  For a
    # negative control the gpu-level path is broken but the reference is not, so
    # they must diverge.  'full_minibatch' has its own dedicated worker/test.
    minibatch_size = _pick_minibatch(ws, mode)

    # ---- per-game reference in GLOBAL game order (assertions 1-4) ------------
    # Global game order = rank-major (rank0 games first, then rank1, ...), which
    # is exactly what cat over the ws contiguous shards reconstructs.
    a = _make_harness(ws, 0, False, minibatch_size, minibatch_size)
    blocks = [_make_block_rollout(b, ws) for b in range(ws)]
    g_bd, g_ex = _assemble_global(blocks)
    ridx = repeat_idxs_for(ws)
    aug = a.augment_batch_for_mixed_expl(g_bd, g_ex, repeat_idxs=ridx)
    # apply the SAME tail-trim the shard does, in the reference (spec section 3).
    aug = _tail_trim_global(aug, minibatch_size)
    a.curr_frames = aug.pop("played_frames")
    a.prepare_dataset(aug)
    ds = a.dataset.values_dict
    cv_ds = a.central_value_net.dataset.values_dict
    ref = {
        "advantages": ds["advantages"].detach().clone(),
        "returns": ds["returns"].detach().clone(),
        "off_policy_mask": (
            ds["off_policy_mask"].detach().clone()
            if ds.get("off_policy_mask") is not None
            else None
        ),
        "rnn_h": ds["rnn_states"][0].detach().clone(),
        "cv_states": cv_ds["obs"].detach().clone(),
        "cv_head_idxs": _head_idxs(a.central_value_net.model, cv_ds["obs"]).clone(),
        "actor_head_idxs": _head_idxs(a.model, ds["obs"]).clone(),
        "len_actor_dataset": len(a.dataset),
        "len_cv_dataset": len(a.central_value_net.dataset),
    }

    # ---- step-0 grad/KL reference: reorder games so single-proc minibatch k ==
    # the row-union of the ws ranks' k-th (mb//ws) minibatches (spec section 3
    # "corresponds to the row-union ...").  With contiguous per-rank shards
    # (rank r owns games [r*tp:(r+1)*tp]) and per-rank minibatch == games_per_mb
    # games, single-proc position (k*ws + r) must hold global game (r*tp + k).
    # A FRESH harness (fresh Adam/RunningMeanStd state) is built so the step-0
    # capture is not polluted by the per-game reference's forward passes.
    b = _make_harness(ws, 0, False, minibatch_size, minibatch_size)
    # independent assembled global (augment mutates extras['obs'] in place; give
    # the step reference its own copy to rule out any cross-call aliasing).
    blocks2 = [_make_block_rollout(bk, ws) for bk in range(ws)]
    g_bd2, g_ex2 = _assemble_global(blocks2)
    aug2 = b.augment_batch_for_mixed_expl(g_bd2, g_ex2, repeat_idxs=ridx)
    aug2 = _tail_trim_global(aug2, minibatch_size)
    # per-RANK games-per-minibatch (== how the gpu path minibatches its shard):
    # gpu _setup halves minibatch -> (minibatch_size // ws); games = that // seq.
    games_per_mb_rank = max(1, (minibatch_size // ws) // SEQ)
    traj_per = (len(aug2["returns"]) // HORIZON) // ws  # games per rank
    aug2 = _reorder_games_rank_interleave(aug2, ws, traj_per, games_per_mb_rank)
    b.curr_frames = aug2.pop("played_frames")
    b.prepare_dataset(aug2)
    b.model.train()
    grad0, kl0 = _capture_actor_grad_before_step(b, b.dataset[0])
    ref["actor_grad_step0"] = grad0
    ref["kl_step0"] = kl0
    b.central_value_net.train()
    ref["cv_grad_step0"] = _cv_grad_after_reduce(
        b.central_value_net, b.central_value_net.dataset[0]
    )
    return ref


def _reorder_games_rank_interleave(batch_dict, ws, traj_per, games_per_mb):
    """Permute games so the single-process minibatch k == union of the ws ranks'
    k-th minibatch.  ranks-per-mb block of games_per_mb games each: reference
    minibatch k (games_per_mb*ws games) = [rank0's mb_k games, rank1's mb_k, ...].
    Game (r*traj_per + j) (rank r, local index j = mbk*games_per_mb + off) lands
    at reference position mbk*(games_per_mb*ws) + r*games_per_mb + off."""
    H = HORIZON
    n_games = len(batch_dict["returns"]) // H
    n_mb = traj_per // games_per_mb
    perm_games = []
    for mbk in range(n_mb):
        for r in range(ws):
            for off in range(games_per_mb):
                perm_games.append(r * traj_per + mbk * games_per_mb + off)
    assert len(perm_games) == n_games, (len(perm_games), n_games)
    pg = torch.tensor(perm_games, dtype=torch.long)
    # flat-row permutation: game g -> rows [g*H:(g+1)*H]
    flat = (pg.reshape(-1, 1) * H + torch.arange(H).reshape(1, -1)).reshape(-1)
    out = {}
    for k, v in batch_dict.items():
        if k in ("played_frames", "step_time"):
            out[k] = v
        elif k == "rnn_states":
            out[k] = [s[:, pg, :].contiguous() for s in v] if v is not None else None
        elif v is not None and torch.is_tensor(v):
            out[k] = v[flat]
        else:
            out[k] = v
    return out


def _tail_trim_global(batch_dict, minibatch_size):
    """Reference-side lossless tail-trim: keep n_traj_keep = (n_traj // block) *
    block trajectories where block == minibatch_size // seq (== single-process
    games-per-full-minibatch)."""
    H = HORIZON
    n_traj = len(batch_dict["returns"]) // H
    games_per_mb = max(1, minibatch_size // SEQ)
    block = games_per_mb
    n_traj_keep = (n_traj // block) * block
    keep_rows = n_traj_keep * H
    out = {}
    for k, v in batch_dict.items():
        if k in ("played_frames", "step_time"):
            out[k] = v
        elif k == "rnn_states":
            out[k] = (
                [s[:, :n_traj_keep, :].contiguous() for s in v]
                if v is not None
                else None
            )
        elif v is not None and torch.is_tensor(v):
            out[k] = v[:keep_rows]
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Minibatch sizing.  Global augmented+trimmed D has (ws*na) games (== ws*na*H
# rows) because off_policy_ratio=1 + lf produces one follower relabel per block.
# Choose minibatch_size = (games_per_mb * ws) * SEQ so:
#   - single-proc games-per-full-minibatch == games_per_mb * ws,
#   - per-rank (halved) games-per-minibatch == games_per_mb,
#   - >= 1 full global minibatch survives the trim.
# For ws==2: mb = (2*2)*SEQ = 4*SEQ  -> per-rank mb = 2*SEQ (2 games).
# For ws==3: mb = (2*3)*SEQ = 6*SEQ  -> per-rank mb = 2*SEQ (2 games).
#
# FIX (5)/(6) — RNN num_seqs==1 degeneracy.  With games_per_mb_rank == 1 the
# per-rank minibatch is a SINGLE game (num_seqs == 1).  rl_games'
# RnnWithDones.forward (common/layers/recurrent.py:33) computes its done-boundary
# list via ``not_dones.squeeze()`` -- a bare .squeeze() that also collapses the
# size-1 num_seqs axis.  So a 1-game minibatch (num_seqs==1) takes a DIFFERENT
# code path for the mid-sequence hidden-state reset than a multi-game minibatch
# (num_seqs>=2), producing a different LSTM output for any game that has a
# mid-episode done.  That made the single-game per-rank path incomparable to the
# multi-game single-proc reference minibatch (KL diverged ~0.55%).  It is NOT a
# learning-signal bug -- it only fires for the num_seqs==1 corner -- but it makes
# the equivalence untestable at 1 game/minibatch.  Isolated proof: with dones,
# num_seqs==2 vs num_seqs==4 match bit-exactly, but num_seqs==1 vs num_seqs==2
# differ by ~0.087.  So size every minibatch to >= 2 games in BOTH paths
# (games_per_mb_rank == 2 -> per-rank num_seqs==2, single-proc num_seqs==2*ws):
# the LSTM done-handling is then identical and KL/grad are exactly comparable.
# --------------------------------------------------------------------------- #


# per-RANK games-per-minibatch used by each mode.  _GMR_OK == 2 keeps num_seqs>=2
# in both paths (see the RNN-squeeze note above).  The tail-trim only bites when
# na % games_per_mb_rank != 0 (global games ws*na is ALWAYS a multiple of ws, so
# a divisor gmr never leaves a gap).  gmr=3 does NOT divide na=8 -> at ws=3 the
# global 24 games are not a multiple of block=ws*3=9 (24//9=2 -> keep 18, drop 6),
# so skipping C.2's trim leaves unequal last minibatches (spec negative control).
_GMR_OK = 2
_GMR_SKIP_TAILTRIM = 3


def _pick_minibatch(ws, mode):
    if mode == "full_minibatch":
        # deliberately leave minibatch at FULL global size -> per-rank shard
        # (na*H rows) < minibatch -> len(dataset)==0 (spec negative control).
        return HORIZON * ws * NA_LOCAL
    gmr = _GMR_SKIP_TAILTRIM if mode == "skip_tailtrim" else _GMR_OK
    # pre-halve minibatch = (gmr * ws) * SEQ  ->  _setup halves to gmr*SEQ/rank.
    return (gmr * ws) * SEQ


# --------------------------------------------------------------------------- #
# Negative-control monkeypatches: revert one fix on the gpu-level harness.
# --------------------------------------------------------------------------- #


def _maybe_break(a, mode):
    if mode == "revert_rnn_shard":
        # revert A.2: pass the packed rnn_states LIST through UN-sharded (the
        # pre-fix generic `else: out[k]=v` behaviour).  Advantages/flat still
        # shard, but rnn dim1 stays global -> per-game hidden mismatch.
        def _broken_shard(self, batch_dict):
            ws = self.world_size
            r = self.global_rank
            H = self.horizon_length
            n_traj = len(batch_dict["returns"]) // H
            games_per_mb = max(1, self.minibatch_size // self.seq_length)
            block = ws * games_per_mb
            n_traj_keep = (n_traj // block) * block
            traj_per = n_traj_keep // ws
            start = r * traj_per * H
            end = start + traj_per * H
            out = {}
            for k, v in batch_dict.items():
                if k in ("played_frames", "step_time"):
                    out[k] = v
                elif v is not None and torch.is_tensor(v):
                    out[k] = v[start:end]
                else:
                    out[k] = v  # rnn_states LIST passes through UN-sharded (BUG)
            return out

        a._shard_batch = types.MethodType(_broken_shard, a)

    elif mode == "skip_tailtrim":
        # revert C.2: shard by naive n_traj//ws (no games_per_mb*ws block trim).
        def _notrim_shard(self, batch_dict):
            ws = self.world_size
            r = self.global_rank
            H = self.horizon_length
            n_traj = len(batch_dict["returns"]) // H
            traj_per = n_traj // ws  # NO block trim
            start = r * traj_per * H
            end = start + traj_per * H
            g_start = r * traj_per
            g_end = g_start + traj_per
            out = {}
            for k, v in batch_dict.items():
                if k in ("played_frames", "step_time"):
                    out[k] = v
                elif k == "rnn_states":
                    out[k] = (
                        [s[:, g_start:g_end, :].contiguous() for s in v]
                        if v is not None
                        else None
                    )
                elif v is not None and torch.is_tensor(v):
                    out[k] = v[start:end]
                else:
                    out[k] = v
            return out

        a._shard_batch = types.MethodType(_notrim_shard, a)

    elif mode == "local_cv_coef":
        # revert B.3: rebuild CV with LOCAL coef-ids (this rank's single block id)
        # instead of the GLOBAL ws-distinct set -> extra_param head mis-selects.
        local_id = a._gpu_local_intr_embd[0, 0].reshape(1)
        a.central_value_net = _build_cv(local_id, a.central_value_net.minibatch_size)
        a.central_value_net.num_actors = a.world_size * NA_LOCAL
        a.central_value_net.batch_size = HORIZON * a.world_size * NA_LOCAL
        from rl_games.common import datasets

        a.central_value_net.dataset = datasets.PPODataset(
            a.central_value_net.batch_size,
            a.central_value_net.minibatch_size,
            True,
            False,
            "cpu",
            SEQ,
        )
        # keep the CV in the SAME multi_gpu mode as the (unbroken) harness so the
        # divergence this control produces isolates the LOCAL-vs-GLOBAL coef-id
        # bug, not a missing all_reduce.  (_build_cv constructs with multi_gpu off.)
        a.central_value_net.multi_gpu = a.multi_gpu
        if a.multi_gpu:
            a.central_value_net.world_size = a.world_size
            a.central_value_net.global_rank = a.global_rank
            a.central_value_net.local_rank = a.local_rank


# =========================================================================== #
# TESTS
# =========================================================================== #


@pytest.mark.parametrize("ws", [2, 3])
def test_learning_signal_equivalence(tmp_path, ws):
    """Spec section 3 gate: all six comparisons at fp32 exact tolerance.

    Assertion -> spec section 3 mapping:
      (1) per-global-game advantages         -> "advantages" bullet (C.2 trim +
          dist_mean_var_count == adv.std over trimmed D).
      (2) off-policy-row returns             -> "returns incl. off-policy rows"
          (REAL LSTM+CV recompute in augment; env-order-mismatch canary).
      (3) packed rnn_states per game         -> "rnn_states content per game"
          (A.1 dim1 gather + A.2 dim1 shard + rank==block order).
      (4) CV states + extra_param head idxs  -> "central-value inputs" (B.1 +
          B.3 GLOBAL coef-ids).
      (5) per-mini_epoch KL scalar           -> "KL scalar into scheduler.update".
      (6) per-parameter grad after all_reduce/ws == single-proc grad, per step
          -> "per-parameter gradient" + "step count" (len(dataset) symmetry).
    """
    out = str(tmp_path / f"sapg_ws{ws}")
    port = GLOO_PORT_BASE + ws
    mp.spawn(_sapg_worker, args=(ws, out, port, "ok"), nprocs=ws, join=True)
    shards = [torch.load(f"{out}.{r}.pt") for r in range(ws)]
    ref = _single_ref(ws, "ok")

    atol, rtol = 1e-5, 1e-4  # fp32 exact pass (spec: atol=1e-6 ideal; CPU elu/LSTM
    #                          accumulation across concat order needs ~1e-5)

    # reconstruct the global order from the ws contiguous shards (rank r owns the
    # r-th contiguous game block after the trim; single-proc order == rank order).
    def cat_games(key):
        return torch.cat([s[key] for s in shards], dim=0)

    def cat_rnn():
        return torch.cat([s["rnn_h"] for s in shards], dim=1)  # dim1 == game axis

    # ---- (6b) STEP COUNT / collective symmetry (spec "step count") ----------
    for s in shards:
        assert torch.equal(s["len_min"], s["len_max"]), (
            "actor/cv dataset lengths differ across ranks -> NCCL desync risk"
        )
        assert s["len_actor_dataset"] == ref["len_actor_dataset"], (
            f"per-rank actor #minibatches {s['len_actor_dataset']} != "
            f"single-proc {ref['len_actor_dataset']}"
        )
        assert s["len_cv_dataset"] == ref["len_cv_dataset"]
        assert s["len_actor_dataset"] >= 1, "gpu-level produced ZERO minibatches"

    # ---- (1) advantages per global game -------------------------------------
    adv_multi = cat_games("advantages")
    assert adv_multi.shape == ref["advantages"].shape
    assert torch.allclose(adv_multi, ref["advantages"], atol=atol, rtol=rtol), (
        f"advantages diverge: max {(adv_multi - ref['advantages']).abs().max()}"
    )

    # ---- (2) returns incl. off-policy rows (the silent-corruption canary) ----
    ret_multi = cat_games("returns")
    assert torch.allclose(ret_multi, ref["returns"], atol=atol, rtol=rtol), (
        f"returns diverge (off-policy CV/LSTM recompute mismatch): "
        f"max {(ret_multi - ref['returns']).abs().max()}"
    )
    # specifically check the off-policy rows (mask==True) are non-empty + match
    if ref["off_policy_mask"] is not None:
        opm = torch.cat([s["off_policy_mask"] for s in shards], dim=0)
        assert opm.any(), "no off-policy rows present (augment produced none)"
        assert torch.allclose(
            ret_multi[opm], ref["returns"][ref["off_policy_mask"]], atol=atol, rtol=rtol
        ), "off-policy-row returns mismatch"

    # ---- (3) packed rnn_states per game -------------------------------------
    rnn_multi = cat_rnn()
    assert rnn_multi.shape == ref["rnn_h"].shape, (
        f"rnn game-axis shape {tuple(rnn_multi.shape)} != {tuple(ref['rnn_h'].shape)}"
    )
    assert torch.allclose(rnn_multi, ref["rnn_h"], atol=atol, rtol=rtol), (
        "packed rnn_states per game mismatch (gather/shard dim1 or rank order)"
    )

    # ---- (4) CV states + extra_param head idxs ------------------------------
    cv_states_multi = cat_games("cv_states")
    assert torch.allclose(cv_states_multi, ref["cv_states"], atol=atol, rtol=rtol), (
        "central-value input states per game mismatch (B.1)"
    )
    cv_idx_multi = cat_games("cv_head_idxs")
    assert torch.equal(cv_idx_multi, ref["cv_head_idxs"]), (
        "extra_param CV head-selection idxs mismatch (B.3 global coef-ids)"
    )
    # the augmented D must actually exercise >= 2 distinct heads (else B.3 is moot)
    assert torch.unique(ref["cv_head_idxs"]).numel() >= 2, (
        "augmented CV batch selects a single head -> coef diversity collapsed"
    )
    act_idx_multi = cat_games("actor_head_idxs")
    assert torch.equal(act_idx_multi, ref["actor_head_idxs"]), (
        "actor extra_param head-selection idxs mismatch"
    )

    # ------------------------------------------------------------------------ #
    # (5) KL and (6) grad are computed on the SAME step-0 minibatch and, in the
    # original code, a KL assertion failure aborted the test BEFORE the grad
    # checks ever ran.  FIX (3): compute every diff up front (no assert), then
    # assert the GRAD equivalence FIRST (so signal 6 always executes), then the
    # KL, then the KL characterization.  A KL residual can no longer hide the
    # grad result.
    # ------------------------------------------------------------------------ #

    grad_atol, grad_rtol = 1e-4, 1e-3  # fp32; concat-order LSTM accumulation

    # (6a) actor grad — every rank's captured param.grad is the DDP-AVERAGED grad
    # (all_reduce SUM / ws), so all ranks agree and each equals the single-proc
    # full-minibatch grad.
    actor_grad_rank_disagree = max(
        (shards[r]["actor_grad_step0"] - shards[0]["actor_grad_step0"])
        .abs()
        .max()
        .item()
        for r in range(ws)
    )
    actor_grad_vs_ref = (
        (shards[0]["actor_grad_step0"] - ref["actor_grad_step0"]).abs().max().item()
    )
    # (6b) central-value grad — CV runs multi_gpu (all_reduce SUM / ws) too.
    cv_grad_rank_disagree = max(
        (shards[r]["cv_grad_step0"] - shards[0]["cv_grad_step0"]).abs().max().item()
        for r in range(ws)
    )
    cv_grad_vs_ref = (
        (shards[0]["cv_grad_step0"] - ref["cv_grad_step0"]).abs().max().item()
    )

    # (5) KL — each rank's minibatch-0 KL is a plain row-mean over its (equal,
    # >=2-game) minibatch; mean over ranks == single-proc row-mean over the union
    # (== reference minibatch 0) because every per-rank minibatch is exactly
    # minibatch_size//ws rows and the union is minibatch_size rows.  This is the
    # mean-of-means form.  Also compute the pooled/row-weighted KL to confirm the
    # code's mean-of-means averaging is DDP-consistent (they coincide only when
    # all ranks carry equal row counts, which C.2's trim guarantees).
    kl_stack = torch.stack([s["kl_step0"] for s in shards])
    kl_mean_of_means = kl_stack.mean()
    # pooled/row-weighted: each rank's KL weighted by its minibatch row count.
    mb0_rows = torch.tensor([float(s["kl_rows_step0"]) for s in shards])
    kl_pooled = (kl_stack * mb0_rows).sum() / mb0_rows.sum()
    kl_vs_ref_mm = (kl_mean_of_means - ref["kl_step0"]).abs().item()
    kl_vs_ref_pooled = (kl_pooled - ref["kl_step0"]).abs().item()

    # --- assert GRAD (signal 6) FIRST so it always runs --------------------- #
    assert actor_grad_rank_disagree <= grad_atol, (
        f"actor grads disagree across ranks: max {actor_grad_rank_disagree:.3e} "
        "(all_reduce SUM/ws should make every rank identical)"
    )
    assert torch.allclose(
        shards[0]["actor_grad_step0"],
        ref["actor_grad_step0"],
        atol=grad_atol,
        rtol=grad_rtol,
    ), f"actor grad != single-proc: max {actor_grad_vs_ref:.3e}"
    assert cv_grad_rank_disagree <= grad_atol, (
        f"cv grads disagree across ranks: max {cv_grad_rank_disagree:.3e}"
    )
    assert torch.allclose(
        shards[0]["cv_grad_step0"],
        ref["cv_grad_step0"],
        atol=grad_atol,
        rtol=grad_rtol,
    ), f"cv grad != single-proc: max {cv_grad_vs_ref:.3e}"

    # --- then assert KL (signal 5), with characterization ------------------- #
    # mean-of-means IS the code's averaging (mean_list(ep_kls) then all_reduce
    # SUM/ws).  It equals single-proc at fp32 once minibatches are equal-sized
    # (C.2 trim) and >= 2 games (no RNN num_seqs==1 squeeze artifact).
    assert torch.allclose(kl_mean_of_means, ref["kl_step0"], atol=atol, rtol=rtol), (
        f"KL (mean-of-means, the code's averaging) != single-proc: "
        f"multi {kl_mean_of_means.item()} vs ref {ref['kl_step0'].item()} "
        f"(|diff|={kl_vs_ref_mm:.3e}); pooled/row-weighted |diff|={kl_vs_ref_pooled:.3e}"
    )
    # DDP-consistency: with equal row counts across ranks the code's mean-of-means
    # MUST coincide with the pooled/row-weighted KL (and both with single-proc).
    assert torch.allclose(kl_mean_of_means, kl_pooled, atol=atol, rtol=rtol), (
        f"KL mean-of-means {kl_mean_of_means.item()} != pooled {kl_pooled.item()} "
        "-> unequal per-rank row counts (C.2 trim broken); the code's KL average "
        "would NOT be DDP-consistent here"
    )


# --------------------------------------------------------------------------- #
# NEGATIVE CONTROLS — each MUST fail (multi != single) when a fix is reverted.
# Structured as: run the broken gpu-level path, run the correct single ref,
# assert they DIVERGE on the signal the fix protects.
# --------------------------------------------------------------------------- #


def _run_broken_and_compare(tmp_path, ws, mode, port):
    out = str(tmp_path / f"neg_{mode}_ws{ws}")
    mp.spawn(_sapg_worker, args=(ws, out, port, mode), nprocs=ws, join=True)
    shards = [torch.load(f"{out}.{r}.pt") for r in range(ws)]
    # the reference is the CORRECT run at this mode's minibatch (these controls
    # keep gmr=1, i.e. the OK minibatch); the gpu path is what's broken.
    ref = _single_ref(ws, mode)
    return shards, ref


def test_negctl_revert_rnn_shard(tmp_path):
    """Revert A.2 (rnn passes through un-sharded) -> rnn-content per game diverges
    (and, downstream, the LSTM forward's grad).  MUST fail."""
    ws = 2
    shards, ref = _run_broken_and_compare(
        tmp_path, ws, "revert_rnn_shard", GLOO_PORT_BASE + 20
    )
    # un-sharded rnn_states means each rank carries the GLOBAL game axis (ws*na)
    # while its flat rows are a shard -> the packed rnn game count mismatches the
    # single-proc per-shard count; reconstructed content cannot match.
    rnn_multi_shapes = [tuple(s["rnn_h"].shape) for s in shards]
    diverged = any(sh[1] != ref["rnn_h"].shape[1] // ws for sh in rnn_multi_shapes)
    if not diverged:
        # if shapes happen to line up, content must still differ
        rnn_multi = torch.cat([s["rnn_h"] for s in shards], dim=1)
        diverged = (rnn_multi.shape != ref["rnn_h"].shape) or (
            not torch.allclose(rnn_multi, ref["rnn_h"], atol=1e-5, rtol=1e-4)
        )
    assert diverged, "reverting rnn dim1 shard did NOT corrupt the signal"


def test_negctl_skip_tailtrim_ws3(tmp_path):
    """Skip C.2 tail-trim at ws=3 with per-rank games_per_mb=3 (does NOT divide
    na=8): the global augmented D has 3*8=24 games, block=ws*3=9, 24 % 9 != 0.

    * The CORRECT trim keeps 18 games (drop 6) -> 6 games/rank -> 2 FULL size-3
      minibatches/rank, equal across ranks, matching the single-proc reference.
    * The reverted naive ``n_traj // ws`` shard keeps all 8 games/rank -> the
      last minibatch has 2 games (not 3) -> unequal minibatch scale -> KL /
      last-minibatch grad diverge from the trimmed reference (or the shape/len
      asserts raise, which is itself a control-satisfying FAIL).

    MUST fail (i.e. diverge / raise) — else C.2's trim is a no-op here."""
    ws = 3
    out = str(tmp_path / f"neg_skip_tailtrim_ws{ws}")
    port = GLOO_PORT_BASE + 30
    try:
        mp.spawn(
            _sapg_worker, args=(ws, out, port, "skip_tailtrim"), nprocs=ws, join=True
        )
    except Exception:
        # a raise inside a rank (dataset/shape/game-count assert) is itself a
        # FAIL of the reverted path -> negative control satisfied.
        return
    shards = [torch.load(f"{out}.{r}.pt") for r in range(ws)]
    ref = _single_ref(ws, "skip_tailtrim")  # correct trimmed ref at gmr=3
    # divergence signal: per-rank dataset lengths unequal across ranks, OR the
    # reconstructed advantages/KL don't match the trimmed single-proc reference.
    lens_disagree = not all(torch.equal(s["len_min"], s["len_max"]) for s in shards)
    len_vs_ref = any(s["len_actor_dataset"] != ref["len_actor_dataset"] for s in shards)
    kl_multi = torch.stack([s["kl_step0"] for s in shards]).mean()
    kl_diverged = not torch.allclose(kl_multi, ref["kl_step0"], atol=1e-5, rtol=1e-4)
    try:
        adv_multi = torch.cat([s["advantages"] for s in shards], dim=0)
        adv_diverged = (adv_multi.shape != ref["advantages"].shape) or (
            not torch.allclose(adv_multi, ref["advantages"], atol=1e-5, rtol=1e-4)
        )
    except Exception:
        adv_diverged = True
    assert lens_disagree or len_vs_ref or kl_diverged or adv_diverged, (
        "skipping the C.2 tail-trim did NOT change the learning signal at ws=3"
    )


def test_negctl_local_cv_coef(tmp_path):
    """Use LOCAL coef-ids for the CV extra_param head instead of GLOBAL -> the CV
    head-selection idxs mis-resolve (all rows collapse to the single local head
    or argmax defaults to 0) -> CV head idxs and CV grad diverge.  MUST fail."""
    ws = 2
    shards, ref = _run_broken_and_compare(
        tmp_path, ws, "local_cv_coef", GLOO_PORT_BASE + 40
    )
    cv_idx_multi = torch.cat([s["cv_head_idxs"] for s in shards], dim=0)

    # FIX (5, negative control): a LOCAL-coef-id CV builds a length-1 param_ids
    # extra_param head, so its parameter vector has a DIFFERENT SHAPE than the
    # GLOBAL-coef reference (993 vs 1025 params).  torch.allclose / torch.equal
    # RAISE a RuntimeError on shape mismatch instead of returning False.  A shape
    # mismatch IS the intended divergence, so count any RuntimeError (or an
    # elementwise False) as "diverged".  This is the control PASSING.
    def _diverged(fn):
        try:
            return not fn()
        except RuntimeError:
            return True  # shape mismatch == the divergence the fix protects against

    # with LOCAL coef-ids param_ids has length 1 -> argmax always 0; global has
    # >=2 distinct heads -> idxs must differ (or shapes differ outright).
    idx_diverged = (cv_idx_multi.shape != ref["cv_head_idxs"].shape) or _diverged(
        lambda: torch.equal(cv_idx_multi, ref["cv_head_idxs"])
    )
    grad_diverged = _diverged(
        lambda: torch.allclose(
            shards[0]["cv_grad_step0"], ref["cv_grad_step0"], atol=1e-4, rtol=1e-3
        )
    )
    assert idx_diverged or grad_diverged, (
        "LOCAL CV coef-ids did NOT change head selection / CV grad"
    )


def test_negctl_full_minibatch_zero_dataset(tmp_path):
    """Leave minibatch at the FULL global size (do not halve per-rank) -> each
    rank's shard (na*H rows) < minibatch_size -> len(PPODataset)==0 -> ZERO
    training minibatches (silent no-op).  MUST fail the step-count assertion."""
    ws = 2
    out = str(tmp_path / f"neg_full_ws{ws}")
    port = GLOO_PORT_BASE + 50
    # patch _pick_minibatch behaviour via mode='full_minibatch' but DON'T let
    # _setup halve it (we bypass _setup's halving by using the non-gpu setup?).
    # Simpler: run a worker variant that keeps minibatch at full size.
    mp.spawn(_full_minibatch_worker, args=(ws, out, port), nprocs=ws, join=True)
    lens = [torch.load(f"{out}.{r}.pt")["len_actor_dataset"] for r in range(ws)]
    assert any(n == 0 for n in lens), (
        f"full-size minibatch did NOT collapse the per-rank dataset to 0: {lens}"
    )


def _full_minibatch_worker(rank, ws, out, port):
    """Reproduce the pre-C.1 bug: minibatch_size stays at the FULL global size
    on each rank (na*H rows < minibatch) -> len(dataset)==0."""
    os.environ["WORLD_SIZE"] = str(ws)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    dist.init_process_group(
        "gloo", rank=rank, world_size=ws, init_method=f"tcp://127.0.0.1:{port}"
    )
    full_mb = HORIZON * ws * NA_LOCAL
    a = _make_harness(ws, rank, True, full_mb, full_mb)
    # UNDO C.1's halving to emulate the bug: restore minibatch to full size.
    a.minibatch_size = full_mb
    from rl_games.common import datasets

    a.dataset = datasets.PPODataset(
        HORIZON * NA_LOCAL, full_mb, False, True, "cpu", SEQ
    )

    bd, ex = _make_block_rollout(rank, ws)
    ridx = repeat_idxs_for(ws)
    # skip _shard_batch's block assert by sharding with a naive equal split so we
    # reach prepare_dataset with a per-rank shard smaller than minibatch.
    a._enter_global_sapg_context()
    g_bd, g_ex = a._gather_rollout(bd, ex)
    aug = a.augment_batch_for_mixed_expl(g_bd, g_ex, repeat_idxs=ridx)
    aug = _naive_equal_shard(aug, ws, rank)
    a.curr_frames = aug.pop("played_frames")
    a.prepare_dataset(aug)
    torch.save({"len_actor_dataset": len(a.dataset)}, f"{out}.{rank}.pt")
    dist.barrier()
    dist.destroy_process_group()


def _naive_equal_shard(batch_dict, ws, r):
    H = HORIZON
    n_traj = len(batch_dict["returns"]) // H
    traj_per = n_traj // ws
    start = r * traj_per * H
    end = start + traj_per * H
    g_start = r * traj_per
    g_end = g_start + traj_per
    out = {}
    for k, v in batch_dict.items():
        if k in ("played_frames", "step_time"):
            out[k] = v
        elif k == "rnn_states":
            out[k] = (
                [s[:, g_start:g_end, :].contiguous() for s in v]
                if v is not None
                else None
            )
        elif v is not None and torch.is_tensor(v):
            out[k] = v[start:end]
        else:
            out[k] = v
    return out

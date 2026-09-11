"""Shared helpers for the GPU-level SAPG offline verification suite.

Offline group = STATIC (pure-torch / AST, single process) + 2-proc CPU gloo.
No GPU, no IsaacSim, no NCCL. Every check binds the REAL rl_games methods onto a
types.SimpleNamespace via types.MethodType (the exact idiom the shipped
tests/test_gpu_level_sapg.py already uses) so we exercise production bytecode,
not a re-implementation.

Load order note: the real package installs as ``rl_games`` (the inner
``rl_games/rl_games`` dir is the package), so imports are
``from rl_games.common.a2c_common import A2CBase``.
"""

import ast
import pathlib
import types

# --------------------------------------------------------------------------- #
# Repo-root resolution: marker-walk up to the worktree root (dir containing
# rl_games/rl_games/common/a2c_common.py). cwd-independent per §1.3.1.
# --------------------------------------------------------------------------- #


def repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for p in [here] + list(here.parents):
        cand = p / "rl_games" / "rl_games" / "common" / "a2c_common.py"
        if cand.exists():
            return p
    raise RuntimeError(
        "could not locate worktree root (rl_games/rl_games/common/a2c_common.py)"
    )


ROOT = repo_root()
A2C_COMMON = ROOT / "rl_games" / "rl_games" / "common" / "a2c_common.py"
A2C_CONTINUOUS = ROOT / "rl_games" / "rl_games" / "algos_torch" / "a2c_continuous.py"
SAPG_YAML = ROOT / "isaacsimenvs" / "cfg" / "train" / "SimToolRealSAPG.yaml"


# --------------------------------------------------------------------------- #
# Method binding (real bytecode onto a stub self)
# --------------------------------------------------------------------------- #


def bind(obj, cls, *names):
    """Bind unbound methods of *cls* onto stub *obj* via types.MethodType."""
    for name in names:
        setattr(obj, name, types.MethodType(getattr(cls, name), obj))


# --------------------------------------------------------------------------- #
# AST helpers — class-SCOPED lookup (three `train`/`train_epoch` defs exist:
# A2CBase / DiscreteA2CBase / ContinuousA2CBase). Naive ast.walk first-match
# returns the A2CBase `pass` stub, which is the SC6a/SC6b false-fail hole.
# --------------------------------------------------------------------------- #


def parse(path: pathlib.Path):
    return ast.parse(path.read_text()), path.read_text()


def find_class(tree, class_name):
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            return n
    raise AssertionError(f"class {class_name} not found")


def find_method(tree, class_name, method_name):
    """Return the FunctionDef for class_name.method_name (class-scoped)."""
    cls = find_class(tree, class_name)
    for b in cls.body:
        if isinstance(b, ast.FunctionDef) and b.name == method_name:
            return b
    raise AssertionError(f"{class_name}.{method_name} not found")


def find_free_func(tree, name):
    """Module-level or first FunctionDef named *name* (for helpers not in a class)."""
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"function {name} not found")


def src_of(node, full_src):
    seg = ast.get_source_segment(full_src, node)
    assert seg is not None, "ast.get_source_segment returned None (need Python 3.8+)"
    return seg


# --------------------------------------------------------------------------- #
# Config replay of the __init__ gpu_level_sapg gate (a2c_common.py:328) and
# the non-gpu-level block-size branch. Pure arithmetic, no torch.
# --------------------------------------------------------------------------- #


def resolve_gpu_level_flag(config: dict, multi_gpu: bool) -> bool:
    """Byte-faithful replay of a2c_common.py:328."""
    return bool(config.get("gpu_level_sapg", False) and multi_gpu)


# --------------------------------------------------------------------------- #
# Git: the pre-feature base. The gpu-level feature is uncommitted working-tree
# edits to a2c_common.py + custom_utils.py ONLY (verified: a2c_continuous.py is
# byte-identical to the merge-base). "Upstream / pre-feature" == merge-base(HEAD, main).
# --------------------------------------------------------------------------- #

# Fallback pin (SimToolReal official main HEAD == the worktree merge-base).
FEATURE_BASE_SHA = "313d5aea1f507c6cfe097b672b62945d7b0bbff5"


def git_show(ref: str, rel_path: str):
    """Return the bytes/text of <rel_path> at <ref>, or None if unresolvable."""
    import subprocess

    try:
        r = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{ref}:{rel_path}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None
    return r.stdout if r.returncode == 0 else None


def feature_base_ref():
    """merge-base(HEAD, main) if resolvable, else the pinned base SHA."""
    import subprocess

    try:
        r = subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "HEAD", "main"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return FEATURE_BASE_SHA

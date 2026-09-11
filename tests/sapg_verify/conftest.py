"""Ensure the worktree root is importable so `tests.sapg_verify._common` and the
installed `rl_games` package both resolve, regardless of pytest invocation cwd."""

import pathlib
import sys

# repo root = the dir that contains rl_games/rl_games/common/a2c_common.py
_here = pathlib.Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "rl_games" / "rl_games" / "common" / "a2c_common.py").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        # rl_games is a package under <root>/rl_games/ (editable install normally
        # handles this; add the inner dir as a fallback for source-tree runs).
        _inner = _p / "rl_games"
        if str(_inner) not in sys.path:
            sys.path.insert(0, str(_inner))
        break

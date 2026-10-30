"""Opt-in 10-node corridor environment.

This environment preserves the legacy 8-node class and adds Eucalyptus and
Cranford as policy nodes.  All ten nodes expose an offset action.  Fixed-duration
Eucalyptus/Cranford phases expose no split actions through the policy mask.
"""
from . import config as C
from .corridor_env import CorridorEnv


class CorridorEnvAll10(CorridorEnv):
    USE_SPLIT_ACTION_MASK = True

    def __init__(self, *args, **kwargs):
        if C.CONTROL_LAYOUT != "all10":
            raise RuntimeError(
                "CorridorEnvAll10 requires RL_CONTROL_LAYOUT=all10 before importing "
                "src.rl.config. Use `python -m src.rl.train_ppo_all10 ...`.")
        super().__init__(*args, **kwargs)


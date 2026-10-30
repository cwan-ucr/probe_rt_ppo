"""Train PPO in the opt-in 10-node control environment.

Offset-only:
    python -m src.rl.train_ppo_all10 --base actuated --arch bilstm_gru ...

Offset + green split:
    RL_SPLIT=1 python -m src.rl.train_ppo_all10 --base actuated \
        --arch bilstm_gru ...

The environment selector must be set before importing config/model modules so
spawned workers and the learner construct identical tensor dimensions.
"""
import os

os.environ["RL_CONTROL_LAYOUT"] = "all10"

from .train_ppo_parallel import main


if __name__ == "__main__":
    main()


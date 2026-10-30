"""Train the Diaz-style detector/queue/absolute-offset MLP comparator.

Example (same demand design and backbone as the probe-controller experiments):
  python -m src.rl.train_diaz_style --backbone coord_actuated --random-demand \
      --workers 32 --traj-per-update 24 --iters 500 --demand-bank-size 2000

All normal parallel-PPO arguments except --arch are forwarded.  The detector
environment and action/reward semantics are fixed here before worker imports.
"""
import os
import sys

os.environ["RL_METHOD"] = "diaz"
os.environ["RL_CONTROL_LAYOUT"] = "all10"
os.environ["RL_RESIDUAL"] = "0"
os.environ["RL_SPLIT"] = "0"
os.environ["RL_OFFSET_DEVIATION_PENALTY"] = "0"
os.environ.setdefault("RL_COLLECTOR", "probe")

from .train_ppo_parallel import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--arch" in argv and argv[argv.index("--arch") + 1] != "mlp":
        raise SystemExit("Diaz-style comparator requires --arch mlp")
    if "--arch" not in argv:
        argv += ["--arch", "mlp"]
    if "--backbone" not in argv and "--base" not in argv:
        argv += ["--backbone", "coord_actuated"]
    if "--no-diff-reward" not in argv:
        argv.append("--no-diff-reward")
    if "--no-warmstart" not in argv:
        argv.append("--no-warmstart")
    if "--run-tag" not in argv:
        argv += ["--run-tag", "diaz_mlp_all10"]
    if "--log-std-init" not in argv:
        # Absolute 0..59-s offsets need broader initial exploration than the
        # near-incumbent residual policy's default (-2.5).
        argv += ["--log-std-init", "-0.5"]
    sys.argv[1:] = argv
    main()

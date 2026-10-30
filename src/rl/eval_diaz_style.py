"""Paired physical evaluation of a train_diaz_style checkpoint."""
import os
import sys

os.environ["RL_METHOD"] = "diaz"
os.environ["RL_CONTROL_LAYOUT"] = "all10"
os.environ["RL_RESIDUAL"] = "0"
os.environ["RL_SPLIT"] = "0"
os.environ["RL_OFFSET_DEVIATION_PENALTY"] = "0"
os.environ.setdefault("RL_COLLECTOR", "probe")

from .eval_policy import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--arch" in argv and argv[argv.index("--arch") + 1] != "mlp":
        raise SystemExit("Diaz-style comparator requires --arch mlp")
    if "--arch" not in argv:
        argv += ["--arch", "mlp"]
    if "--backbone" not in argv and "--base" not in argv:
        argv += ["--backbone", "coord_actuated"]
    if "--tag" not in argv:
        argv += ["--tag", "diaz_mlp_all10"]
    sys.argv[1:] = argv
    main()

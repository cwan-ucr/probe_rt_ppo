"""Evaluate a checkpoint trained with the 10-node corridor environment."""
import os

os.environ["RL_CONTROL_LAYOUT"] = "all10"

from .eval_policy import main


if __name__ == "__main__":
    main()


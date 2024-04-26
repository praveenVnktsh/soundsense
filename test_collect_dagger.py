import sys
import numpy as np
import time

import yaml
import torch
from robot_node_dagger import RobotNode


if __name__ == "__main__":
    from models.baselines.mulsa.inference import MULSAInference

    # 86 - move left
    # 30 - move right
    model_root = "/home/hello-robot/soundsense/soundsense/models/baselines/mulsa/trained_models_new/"
    model_name = 'sorting_imi_vg_simple_seqlen_1_mha_spec04-22-04:18:40'
    model_root += model_name
    model_root += '/'
    ckptname = 'epoch=49-step=16050.ckpt'
    print("Loading hparams from ", model_root + "hparams.yaml")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    robot = RobotNode(
        config_path = model_root + "hparams.yaml",
        model_name = model_name + '/' + ckptname,
    )

    robot.run_loop(True)
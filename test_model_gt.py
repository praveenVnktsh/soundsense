import sys
import numpy as np
import time

import yaml
import torch
from robot_node_test_with_gt import RobotNode


if __name__ == "__main__":
    from models.baselines.mulsa.inference import MULSAInference

    # 86 - move left
    # 30 - move right
    model_root = "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/lightning_logs/"
    model_root += "sorting_imi_vg_ag_lstm_seqlen_3_mha_spec04-22-19:26:28"
    model_root += '/'
    model_name = 'last.ckpt'
    # model_name = '04-09-15:48:16-v1.ckpt'
    print("Loading hparams from ", model_root + "hparams.yaml")
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    model = MULSAInference(
        config_path = model_root + "hparams.yaml",
    )

    model.load_state_dict(
        torch.load(
            model_root + model_name,
            map_location=torch.device("cuda"),
        )['state_dict']
    )
    model.eval()

    robot = RobotNode(
        config_path = model_root + "hparams.yaml",
        model = model, 
        testing = True,
        run_id = sys.argv[1]
        
    )

    robot.run_loop(True)
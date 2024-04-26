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
    model_root += "Fri-04-26-10:17sorting_imi_vg_ag_simple_seqlen_1_mha_spec_audio_len_5_num_stacks_6"
    model_root += '/'
    model_name = 'last.ckpt'
    # model_name = ';astckpt'
    # model_name = '04-09-15:48:16-v1.ckpt'
    print("Loading hparams from ", model_root + "hparams.yaml")
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    model = MULSAInference(
        config_path = model_root + "hparams.yaml",
    ).cuda()

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
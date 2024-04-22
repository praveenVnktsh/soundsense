import sys
import numpy as np
import rospy
import time
from audio_common_msgs.msg import AudioDataStamped, AudioData
from robot_node import RobotNode
import yaml
import torch
from robot_node import RobotNode


if __name__ == "__main__":
    from models.baselines.dummy.model import LitModel 
    from models.baselines.mulsa.inference import MULSAInference
    # from models.baselines.dummy.model import LitModel as DummyModel
    # JUST IMPORT THE CORRECT MODEL FROM HERE BRO!

    
    
    # action_path = '/home/hello-robot/soundsense/soundsense/stretch/data/data_two_cups/3/actions.json'
    # model = LitModel(action_path = action_path)
    # model.eval()
    # If using MULSA, change out to out[0] in inference.py line 99
    
    model_root = "/home/hello-robot/soundsense/soundsense/models/baselines/mulsa/trained_models/"
    # model_root += "mulsa_cnn_unimodal_full_task04-07-12:35:14"
    # model_root += "imi_vg_lstm_seqlen_10_mha_spec04-20-07:32:19"
    # model_root += "imi_vg_ag_lstm_seqlen_10_mha_spec04-19-21:41:47"
    model_root += 'c'
    # model_root += 'sorting_imi_vg_lstm_seqlen_10_mha_spec04-21-15:46:27'
    model_root += '/'
    model_name = 'last.ckpt'
    # model_name = '04-09-15:48:16-v1.ckpt'
    print("Loading hparams from ", model_root + "hparams.yaml")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    model = MULSAInference(
        config_path = model_root + "hparams.yaml",
    )

    model.load_state_dict(
        torch.load(
            model_root + model_name,
            map_location=torch.device("cpu"),
        )['state_dict']
    )
    # model.load_from_checkpoint(
    #     model_root + model_name,
    # )

    model.eval()

    robot = RobotNode(
        config_path = model_root + "hparams.yaml",
        model = model, 
        testing = True,
        
    )

    robot.run_loop(True)

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

    rospy.init_node("test_model")
    
    # action_path = '/home/hello-robot/soundsense/soundsense/stretch/data/data_two_cups/3/actions.json'
    # model = LitModel(action_path = action_path)
    # model.eval()
    # If using MULSA, change out to out[0] in inference.py line 99
    
    model_root = "/home/hello-robot/soundsense/soundsense/models/baselines/mulsa/trained_models_new/"
    # model_root += "mulsa_cnn_unimodal_full_task04-07-12:35:14"
    # model_root += "imi_vg_lstm_seqlen_10_mha_spec04-20-07:32:19"
    # model_root += "sorting_imi_vg_lstm_seqlen_3_mha_spec04-21-21:13:43"
    # model_root += "sorting_imi_vg_lstm_seqlen_3_spec04-22-00:48:38"
    model_root += "sorting_imi_vg_ag_simple_seqlen_1_spec04-22-17:19:32"
    model_root += '/'
    model_name = 'epoch=49-step=16050.ckpt'
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
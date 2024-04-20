import sys
import numpy as np
# import rospy
import time
# from audio_common_msgs.msg import AudioDataStamped, AudioData
# from robot_node import RobotNode
import yaml
import torch
from robot_node_test_with_gt import RobotNode


if __name__ == "__main__":
    from models.baselines.dummy.model import LitModel 
    from models.baselines.mulsa.inference import MULSAInference
    # from models.baselines.dummy.model import LitModel as DummyModel
    # JUST IMPORT THE CORRECT MODEL FROM HERE BRO!

    # rospy.init_node("test_model")
    
    # action_path = '/home/hello-robot/soundsense/soundsense/stretch/data/data_two_cups/3/actions.json'
    # model = LitModel(action_path = action_path)
    # model.eval()
    # If using MULSA, change out to out[0] in inference.py line 99
    
    # model_root = "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/old_logs/"
    # model_root = "/home/hello-robot/soundsense/soundsense/models/baselines/mulsa/trained_models/"
    model_root = '/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/lightning_logs/'
   
    model_root += 'imi_vg_lstm_seqlen_10_mha_spec04-19-21:02:40'
    model_root += '/'
    model_name = 'last.ckpt'
    # model_name = '04-09-15:48:16-v1.ckpt'
    print("Loading hparams from ", model_root + "hparams.yaml")

    
    model = MULSAInference(
        config_path = model_root + "hparams.yaml",
    ).cuda()

    model.load_state_dict(
        torch.load(
            model_root + model_name,
            map_location=torch.device("cuda"),
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

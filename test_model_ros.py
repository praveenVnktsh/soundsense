import sys
import numpy as np
import rospy
import time
from audio_common_msgs.msg import AudioDataStamped, AudioData
from robot_node import RobotNode
import yaml
import torch
from robot_node_separate import RobotNode


if __name__ == "__main__":

    rospy.init_node("test_model")
    
    
    model_root = "/home/hello-robot/soundsense/soundsense/models/baselines/mulsa/trained_models/"
    model_root += "sorting_imi_vg_ag_simple_seqlen_1_mha_spec04-22-15:08:58"
    model_root += '/'
    model_name = 'last.ckpt'
    # model_name = '04-09-15:48:16-v1.ckpt'
    print("Loading hparams from ", model_root + "hparams.yaml")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    robot = RobotNode(
        config_path = model_root + "hparams.yaml",
        testing = True,
        model = model,
    )

    robot.run_loop(True)
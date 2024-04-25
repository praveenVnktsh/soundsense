import sys
import numpy as np
import rospy
import time
from audio_common_msgs.msg import AudioDataStamped, AudioData
from robot_node import RobotNode
import yaml
import torch
from robot_node_separate import RobotNode
# - [CNN](sorting_imi_vg_simple_seqlen_1_spec04-22-07:46:29)
# - [CNN+LSTM](sorting_imi_vg_lstm_seqlen_3_spec04-22-00:48:38)
# - [MHA](sorting_imi_vg_simple_seqlen_1_mha_spec04-22-04:18:40)

# Multimodal
# - [CNN](sorting_imi_vg_ag_simple_seqlen_1_spec04-22-17:19:32)
# - [CNN+LSTM](sorting_imi_vg_ag_lstm_seqlen_3_spec04-22-21:39:20)
# - [MHA](sorting_imi_vg_ag_simple_seqlen_1_mha_spec04-22-15:08:58)


# Proposed

# - [MHA+LSTM](sorting_imi_vg_ag_lstm_seqlen_3_mha_spec04-22-19:26:28)
# - [MHA+LSTM Unimodal (not used)](sorting_imi_vg_lstm_seqlen_3_mha_spec04-21-21:13:43)

if __name__ == "__main__":

    rospy.init_node("test_model")
    
    
    model_root = "/home/hello-robot/soundsense/soundsense/models/baselines/mulsa/trained_models_new/"
    # model_root += "sorting_imi_vg_ag_lstm_seqlen_3_spec04-22-21:39:20"
    model_name = 'sorting_imi_vg_ag_simple_seqlen_1_mha_spec04-22-15:08:58'
    model_root += model_name
    model_root += '/'
    ckptname = 'last.ckpt'
    # model_name = '04-09-15:48:16-v1.ckpt'
    print("Loading hparams from ", model_root + "hparams.yaml")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    robot = RobotNode(
        config_path = model_root + "hparams.yaml",
        model_name = model_name + '/' + ckptname,
        testing = True,
        model = model,
    )

    robot.run_loop(True)
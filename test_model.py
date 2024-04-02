import numpy as np
import rospy
import time
from audio_common_msgs.msg import AudioDataStamped, AudioData
from robot_node_test_with_gt import RobotNode

if __name__ == "__main__":
    from models.baselines.dummy.model import LitModel 
    from models.baselines.mulsa.inference import MULSAInference
    # from models.baselines.dummy.model import LitModel as DummyModel
    # JUST IMPORT THE CORRECT MODEL FROM HERE BRO!

    rospy.init_node("test_model")
    is_unimodal = True # flag to enable audio capture and recording.
    # action_path = '/home/hello-robot/soundsense/soundsense/stretch/data/data_two_cups/3/actions.json'
    # model = LitModel(action_path = action_path)
    # model.eval()
    # If using MULSA, change out to out[0] in inference.py line 99
    model_path = "/home/hello-robot/soundsense/soundsense/models/baselines/mulsa/test_models/03-31-18:34:39.ckpt"
    model = MULSAInference.load_from_checkpoint(model_path)
    model.eval()

    robot = RobotNode(
        config_path='config/test.yaml',
        model= model, 
        is_unimodal = is_unimodal
    )
    robot.run_loop()

import numpy as np
import rospy
import time
from audio_common_msgs.msg import AudioDataStamped, AudioData
from robot_node import RobotNode

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
    model_path = "/home/hello-robot/soundsense/soundsense/stretch/models/baselines/mulsa/weights/unimodal/simple_task/03-28-16:48:44-jobid=0-epoch=2-step=2634.ckpt"
    model = MULSAInference.load_from_checkpoint(model_path)
    model.eval()

    robot = RobotNode(
        config_path='config/test.yaml',
        model= model, 
        is_unimodal = is_unimodal
    )
    robot.run_loop()
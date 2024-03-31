import numpy as np
import torch
import stretch_body.robot
import time
import cv2
from stretch.robot_node import boot_robot, get_image, run_loop
import rospy
import time
from audio_common_msgs.msg import AudioDataStamped, AudioData
import wave
from robot_node import RobotNode
if __name__ == "__main__":
    from models.baselines.dummy.model import LitModel 
    # from models.baselines.dummy.model import LitModel as DummyModel
    # JUST IMPORT THE CORRECT MODEL FROM HERE BRO!

    rospy.init_node("test_model")


    is_unimodal = True # flag to enable audio capture and recording.
    action_path = '/home/hello-robot/soundsense/soundsense/stretch/data/data_two_cups/3/actions.json'
    model = LitModel(action_path = action_path)
    model.eval()
    robot = RobotNode(model= model, is_unimodal = is_unimodal)
    
    
    rospy.sleep(10)

    # hz = 16000
    # w = wave.open('temp.wav', 'wb')
    # w.setnchannels(1)
    # w.setsampwidth(2)
    # w.setframerate(hz)
    # w.setnframes(n_frames)
    # w.writeframes(b''.join(buffer))
    # w.close()
    
    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print("exiting")
    


    # run_loop(model)

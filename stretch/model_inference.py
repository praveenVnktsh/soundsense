import numpy as np
import torch
import stretch_body.robot
import time
import cv2
import sys
import copy
# from soundsense.baselines.cnn.model import LitModel
from models.baselines.cnnlstm.model import CNNLSTMWithResNetForActionPrediction
from models.baselines.mulsa.src.inference import MULSAInference

sys.path.append("")

lookup = {
    0: [1., 0.5, 0.5, 0.5, 0.5],
    1: [0., 0.5, 0.5, 0.5, 0.5],
    2: [0.5, 0., 0.5, 0.5, 0.5],
    3: [0.5, 1., 0.5, 0.5, 0.5],
    4: [0.5, 0.5, 1., 0.5, 0.5],
    5: [0.5, 0.5, 0., 0.5, 0.5],
    6: [0.5, 0.5, 0.5, 1., 0.5],
    7: [0.5, 0.5, 0.5, 0., 0.5],
    8: [0.5, 0.5, 0.5, 0.5, 0.],
    9: [0.5, 0.5, 0.5, 0.5, 1.],
    10: [0.5, 0.5, 0.5, 0.5, 0.5],
}

def convert_to_action(model, inp):
    # action_space = [del_extension, del_height, del_lateral, del_roll, del_gripper]
    limits = [
        [-0.05, 0.05], # extension
        [-0.05, 0.05], # base
        [-50, 50],      # gripper
        [-0.05, 0.05], # lift
        [-10*np.pi/180, 10 * np.pi/180] # roll
    ]
    outputs = model(inp) # 11x1
    action_idx = torch.argmax(outputs)
    lookup_copy = copy.deepcopy(lookup)
    outputs = lookup_copy[int(action_idx)]
    print("before norm argmax:", int(action_idx), "output:",outputs)
    print("lookup, lookup[id]",lookup,lookup[int(action_idx)])
    for i in range(len(limits)):
        outputs[i] = outputs[i] * (limits[i][1] - limits[i][0]) + limits[i][0]
    return outputs

# Model:
# extension+, extension-, base-, base+, gripper+, gripper-, lift+, lift-, roll-, roll+, no_action
# Map to:
# extension, base, gripper, lift, roll

# 0: [1, 0.5, 0.5, 0.5, 0.5]
# 1: [0, 0.5, 0.5, 0.5, 0.5]
# 2: [0.5, 0, 0.5, 0.5, 0.5]
# 3: [0.5, 1, 0.5, 0.5, 0.5]
# 4: [0.5, 0.5, 1, 0.5, 0.5]
# 5: [0.5, 0.5, 0, 0.5, 0.5]
# 6: [0.5, 0.5, 0.5, 1, 0.5]
# 7: [0.5, 0.5, 0.5, 0, 0.5]
# 8: [0.5, 0.5, 0.5, 0.5, 0]
# 9: [0.5, 0.5, 0.5, 0.5, 1]

# 0.1, 0.2, 0.1, 0.3, -0.1, 0.3, 0.5, 0.3, 0.2, 0.2, 0.1
# --------| --------| ---------| --------| --------| ---
def get_image(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return torch.tensor(np.expand_dims(np.array(frame).transpose(2, 0, 1), (0, 1))).float()


def run_loop(model):
    is_run = True
    start_time = time.time()
    
    cap = cv2.VideoCapture(6)
    
    loop_rate = 1
    loop_start_time = time.time()

    while is_run:
        if time.time() - start_time > 10:
            is_run = False

        if time.time() - loop_start_time > 1/loop_rate:
            loop_start_time = time.time()
            frame = get_image(cap)
            if frame is not None:
                action = convert_to_action(model, frame)
                execute_action(r, action)
                print("action:  ",action)
            else:
                print("No frame")
                is_run = False

    print("Ending run loop")
    cap.release()

def execute_action(r: stretch_body.robot.Robot , action):
    # action = [extension, base, gripper, lift, roll]
    r.arm.move_by(action[0])
    r.lift.move_by(action[3])
    r.base.translate_by(action[1])      
    r.end_of_arm.move_by('wrist_roll', action[4])
    r.end_of_arm.move_by('stretch_gripper', action[2])
    r.push_command()
    return True

if __name__ == "__main__":

    r = stretch_body.robot.Robot()
    if not r.startup():
        print("Failed to start robot")
        exit() # failed to start robot!
    if not r.is_homed():
        print("Robot is not calibrated. Do you wish to calibrate? (y/n)")
        if input() == "y":
            r.home()
        else:
            print("Exiting...")
            exit()

    r.lift.move_to(0.9)
    r.arm.move_to(0.0)
    r.end_of_arm.move_to('wrist_yaw', 0.0)
    r.end_of_arm.move_to('wrist_pitch', 0.0)
    r.end_of_arm.move_to('wrist_roll', 0.0)
    r.end_of_arm.move_to('stretch_gripper', 50)
    r.push_command()
    r.lift.wait_until_at_setpoint()
    r.arm.wait_until_at_setpoint()
    


    # model = MULSAInference.load_from_checkpoint("/home/hello-robot/soundsense/soundsense/stretch/models/baselines/mulsa/03272024_205921_last.ckpt")
    model = CNNLSTMWithResNetForActionPrediction.load_from_checkpoint("/home/hello-robot/soundsense/soundsense/stretch/models/baselines/cnnlstm/epoch=28-step=8352.ckpt")

    run_loop(model)
    r.stop()

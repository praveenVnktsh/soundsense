import numpy as np
import torch
import stretch_body.robot
import time
import cv2
from soundsense.baselines.playbyear.core.drq_memory import DRQAgent

def convert_to_action(agent, inp):
    # action_space = [del_extension, del_height, del_lateral, del_roll, del_gripper]
    limits = [
        [-0.05, 0.05],
        [-0.05, 0.05],
        [-0.05, 0.05],
        [-10*np.pi/180, 10 * np.pi/180],
        [20, 100]
    ]
    outputs = agent.actor(inp).mean # 11x1
    print(outputs.shape)
    for i in range(len(limits)):
        outputs[i] = outputs[i] * (limits[i][1] - limits[i][0]) + limits[i][0]

    return outputs


def get_image(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


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
                print(action)
            else:
                print("No frame")
                is_run = False

    print("Ending run loop")
    cap.release()

def execute_action(r: stretch_body.robot.Robot , action):
    # action_space = [del_extension, del_height, del_lateral, del_roll, del_gripper]
    r.arm.move_by(action[0])
    r.lift.move_by(action[1])
    r.base.translate_by(action[2])
    r.end_of_arm.move_by('wrist_roll', action[3])
    r.end_of_arm.move_by('stretch_gripper', action[4])
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

    r.lift.move_to(0.7)
    r.arm.move_to(0.0)
    r.end_of_arm.move_to('wrist_yaw', 0.0)
    r.end_of_arm.move_to('wrist_pitch', 0.0)
    r.end_of_arm.move_to('wrist_roll', 0.0)
    r.end_of_arm.move_to('stretch_gripper', 50)
    r.push_command()
    r.lift.wait_until_at_setpoint()
    r.arm.wait_until_at_setpoint()
    


    model = CNN()
    run_loop(model)
    r.stop()

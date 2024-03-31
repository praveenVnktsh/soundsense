import numpy as np
import torch
import stretch_body.robot
import time
import cv2
from model import LitModel

def get_image(cap):
    h, w = 224, 224
    ret, frame = cap.read()
    frame = cv2.resize(frame, (h, w))
    if not ret:
        return None
    return frame


def run_loop(model):
    is_run = True
    start_time = time.time()
    
    cap = cv2.VideoCapture(6)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    loop_rate = 10
    loop_start_time = time.time()
    history = []
    n_stack = 10
    while is_run:
        # if time.time() - start_time > 10:
        #     is_run = False
        frame = get_image(cap)

        if len(history) > 0:
            history[-1] = frame.copy()
        if time.time() - loop_start_time > 1/loop_rate:
            loop_start_time = time.time()
            
            # cv2.imshow('frame', frame)
            
            
            history.append(frame)
            if frame is not None:
                if len(history) < n_stack:
                    continue
                if len(history) > n_stack:
                    history = history[-n_stack:]

                stacked_inp = np.hstack(history)
                
                cv2.imshow('stacked', stacked_inp)
                cv2.waitKey(1)
                execute_action(r, model, history)
            else:
                print("No frame")
                is_run = False

    print("Ending run loop")
    cap.release()

def execute_action(r: stretch_body.robot.Robot, model, inp):
    # lift, extension, lateral, roll, gripper
    # action_space = [del_extension, del_height, del_lateral, del_roll, del_gripper]
    # action[0] = np.clip(action[0], -0.01, 0.01)
    outputs = model(inp) # 11 dimensional
    # w - extend arm
    # s - retract arm
    # a - move left
    # d - move right

    # i - lift up
    # k - lift down

    # l - roll right
    # j - roll left

    # m - close gripper
    # n - open gripper
    mapping = {
        'w': 0, 
        's': 1,
        'a': 2,
        'd': 3,
        'n': 4,
        'm': 5,
        'i': 6,
        'k': 7,
        'j': 8,
        'l': 9,
    }

    action_idx = torch.argmax(outputs).item()
    print("Performing action: ", action_idx, outputs)
    big = 0.05
    movement_resolution = {
        0: big,
        1: -big,
        2: big,
        3: -big,
        4: 100,
        5: 30,
        6: big,
        7: -big,
        8: 15 * np.pi/180,
        9: -15 * np.pi/180,
    }
    if action_idx in [0, 1]:
        r.arm.move_by(movement_resolution[action_idx])
    elif action_idx in [2, 3]:
        r.base.translate_by(movement_resolution[action_idx])
    elif action_idx in [4, 5]:
        r.end_of_arm.move_by('stretch_gripper', movement_resolution[action_idx])
    elif action_idx in [6, 7]:
        r.end_of_arm.move_by('wrist_roll', movement_resolution[action_idx])
    elif action_idx in [8, 9]:
        r.lift.move_by(movement_resolution[action_idx])
    r.push_command()
    # time.sleep(1)

    return True

if __name__ == "__main__":
    import time
    r = stretch_body.robot.Robot()
    model = LitModel.load_from_checkpoint("/home/hello-robot/soundsense/soundsense/stretch/models/baselines/cnn/e246_10.ckpt", audio =False, n_stacked = 10)
    model.eval()
    
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
    r.arm.move_to(0.1)
    r.end_of_arm.move_to('wrist_yaw', 0.0)
    r.end_of_arm.move_to('wrist_pitch', 0.0)
    r.end_of_arm.move_to('wrist_roll', 0.0)
    r.end_of_arm.move_to('stretch_gripper', 0)
    r.push_command()
    r.lift.wait_until_at_setpoint()
    r.arm.wait_until_at_setpoint()
    time.sleep(5)
    while True:
        run_loop(model)
        print("Loop ended")
        break

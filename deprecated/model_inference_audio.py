import datetime
import os
import wave
import rospy
import numpy as np
import torch
import stretch_body.robot
import time
import cv2
import sys
import copy
# from soundsense.baselines.cnn.model import LitModel
from models.baselines.cnnlstm.model import CNNLSTMWithResNetForActionPrediction
from models.baselines.mulsa.inference import MULSAInference
import pyaudio, alsaaudio

sys.path.append("")

HISTORY_LEN = 6
AUDIO_HISTORY_LEN = 0.01 # seconds

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

limits = [
        [-0.05, 0.05], # extension
        [-0.05, 0.05], # base
        [-50, 50],      # gripper
        [-0.05, 0.05], # lift
        [-10*np.pi/180, 10 * np.pi/180] # roll
    ]

def convert_to_action(model, inp):
    # action_space = [del_extension, del_height, del_lateral, del_roll, del_gripper]
    print("before model inp. inp shape", inp.size())
    outputs = model(inp) # 11x1
    print("output shape", outputs[0].size())
    action_idx = torch.argmax(outputs[0])
    lookup_copy = copy.deepcopy(lookup)
    outputs = lookup_copy[int(action_idx)]
    print("before norm argmax:", int(action_idx), "output:",outputs)
    # print("lookup, lookup[id]",lookup,lookup[int(action_idx)])
    for i in range(len(limits)):
        outputs[i] = outputs[i] * (limits[i][1] - limits[i][0]) + limits[i][0]
    return outputs

def convert_to_action_w_audio(model, inp, audio_inp):
    '''
    audio_inp: numpy array of audio data
    '''
    # Convert audio to wav
    # run_id = datetime.datetime.now().strftime('%s%f')
    # pathFolder = 'data/' + run_id + '/'
    # os.makedirs(pathFolder, exist_ok= True)
    # os.makedirs(pathFolder + 'video/', exist_ok= True)
    # # filename = datetime.datetime.now().strftime('%s%f')
    # filename = "audio_3s"
    # path = pathFolder + f"{filename}.wav"
    # w = wave.open(path, 'w')
    # w.setnchannels(1)
    # w.setsampwidth(2)
    # w.setframerate(48000)
    # w.writeframes(audio_inp)
    # w.close()
    # audio_inp = np.expand_dims(audio_inp,0)
    audio_gripper = [
        x for x in audio_inp if x is not None
    ]
    # print("ag", torch.tensor(audio_gripper).shape) #(434176, 2)
    audio_gripper = torch.as_tensor(np.stack(audio_gripper, 0))
    # print("ag1", audio_gripper.shape) #(434176, 2)
    # audio_gripper = (audio_gripper.T[0,:]).reshape(1,-1)
    audio_gripper = (audio_gripper).reshape(1,-1)
    audio_gripper = audio_gripper.unsqueeze(0)

    print("before model inp")
    outputs = model((inp, audio_gripper)) # 11x1
    print("output shape", outputs[0].size())



    action_idx = torch.argmax(outputs[0])
    lookup_copy = copy.deepcopy(lookup)
    outputs = lookup_copy[int(action_idx)]
    print("before norm argmax:", int(action_idx), "output:",outputs)
    # print("lookup, lookup[id]",lookup,lookup[int(action_idx)])
    for i in range(len(limits)):
        outputs[i] = outputs[i] * (limits[i][1] - limits[i][0]) + limits[i][0]
    return outputs



def get_image(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return torch.tensor(np.expand_dims(np.array(frame).transpose(2, 0, 1), (0, 1))).float()

def get_image_history(cap, history, history_len):
    while True:
        ret, frame = cap.read()
        # print(type(frame))
        if not ret:
            return None
        # print("frame before append", torch.tensor(frame).unsqueeze(0).unsqueeze(0).permute(0,1,4,2,3).size())
        history = torch.cat((history, torch.tensor(frame).unsqueeze(0).unsqueeze(0).permute(0,1,4,2,3)), 1)
        # print("history after append", history.size())

        if history.size(1) > history_len:
            history = history[:,1:,:,:,:]
        
        if history.size(1) == history_len:
            break
    
    return history.float()
        
# def audio_start():
#     print("Starting audio loop")
#     global inp
inp = alsaaudio.PCM(1)
inp.setchannels(1)
inp.setrate(48000)
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
inp.setperiodsize(1024)


def get_audio_history(audio_history, audio_history_len):
    # global inp
    print("start get_audio_history")
    while True:
        print("looping", len(audio_history))
        l, data = inp.read()
        a = np.frombuffer(data, dtype=np.int16)
        audio_history = np.append(audio_history, a)
        if len(audio_history) >= audio_history_len * 48000:
            audio_history = audio_history[-int(audio_history_len * 48000):]
            break
    audio_history = np.array(audio_history)
    return audio_history


def run_loop(model):
    # audio_start()
    is_run = True
    start_time = time.time()
    
    cap = cv2.VideoCapture(6)
    
    loop_rate = 1
    loop_start_time = time.time()
    history = torch.zeros(size=(1,1,3,720,1280))
    audio_history = np.array([])
    print("start loop")
    while is_run:
        # if time.time() - start_time > 10:
        #     is_run = False
        print("getting images...",end=" ")
        history = get_image_history(cap, history, HISTORY_LEN)
        print("done")
        print("getting audio...",end=" ")
        audio_history = get_audio_history(audio_history, AUDIO_HISTORY_LEN)
        print("done")

        if time.time() - loop_start_time > 1/loop_rate:
            loop_start_time = time.time()
            # history = get_image(cap)
            # history = get_image_history(cap, history, HISTORY_LEN)
            if history is not None:
                action = convert_to_action_w_audio(model, history, audio_history)
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
    

    model = MULSAInference.load_from_checkpoint("/home/hello-robot/soundsense/soundsense/stretch/models/baselines/mulsa/03272024_205921_last.ckpt")
    
    # model = MULSAInference.load_from_checkpoint("/home/hello-robot/soundsense/soundsense/stretch/models/baselines/mulsa/exp03282024_16484_last.ckpt")
    # model = CNNLSTMWithResNetForActionPrediction.load_from_checkpoint("/home/hello-robot/soundsense/soundsense/stretch/models/baselines/cnnlstm/epoch=28-step=8352.ckpt")
    run_loop(model)
    r.stop()

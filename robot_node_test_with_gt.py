import numpy as np
import torch
import stretch_body.robot
import time
import cv2
import yaml
import rospy
from audio_common_msgs.msg import AudioDataStamped, AudioData
import glob
class RobotNode:
    def __init__(self, config_path, model, is_unimodal = False):
        # self.r = stretch_body.robot.Robot()
        # self.boot_robot()
        
        with open(config_path) as info:
            params = yaml.load(info.read(), Loader=yaml.FullLoader)

        self.image_shape = params['camera_inp_h_w']
        self.bgr_to_rgb = params['bgr_to_rgb']

        self.hz = params['audio_hz']
        self.audio_n_seconds = params['audio_history_seconds']
        cam = params['camera_id']
        self.n_stack_images = params['camera_stack_images']
        self.history = {
            'audio': [],
            'video': [],
        }
        if not is_unimodal:
            audio_sub = rospy.Subscriber('/audio/audio', AudioData, self.callback)
            print("Waiting for audio data...")
            rospy.wait_for_message('/audio/audio', AudioData, timeout=10)
        self.idx = 0
        self.model = model
        self.images = sorted(glob.glob('/home/hello-robot/soundsense/soundsense/stretch/data/data_simple/6/video/*.png'))
    
    def callback(self, data):
        audio = np.frombuffer(data.data, dtype=np.uint8)
        self.history['audio'].append(audio)
        if len(self.history['audio']) > self.audio_n_seconds * self.hz:
            self.history['audio'] = self.history['audio'][-self.audio_n_seconds * self.hz:]

    def boot_robot(self,):
        r = self.r
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
        r.arm.move_to(0.3)
        r.end_of_arm.move_to('wrist_yaw', 0.0)
        r.end_of_arm.move_to('wrist_pitch', 0.0)
        r.end_of_arm.move_to('wrist_roll', 0.0)
        r.end_of_arm.move_to('stretch_gripper', 100)
        r.push_command()
        r.lift.wait_until_at_setpoint()
        r.arm.wait_until_at_setpoint()
        time.sleep(5)
        print("Robot ready to run model.")
    
    def get_image(self):
        h, w = self.image_shape 
        self.idx += 1
        self.idx = self.idx % len(self.images)
        path = self.images[self.idx]
        frame = cv2.imread(path)
        frame = cv2.resize(frame, (h, w))
        # frame = np.random.randint(0, 255, (h, w, 3)).astype(np.uint8)
        if self.bgr_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def run_loop(self, visualize = False):
        is_run = True
        start_time = time.time()
        loop_rate = 10
        loop_start_time = time.time()
        
        n_stack = self.n_stack_images * self.audio_n_seconds
        while is_run:
            frame = self.get_image()
            if len(self.history['video']) > 0:
                self.history['video'][-1] = frame.copy()
            if time.time() - loop_start_time > 1/loop_rate:
                loop_start_time = time.time()
                
                self.history['video'].append(frame)
                if frame is not None:
                    if len(self.history['video']) < n_stack:
                        continue
                    if len(self.history['video']) > n_stack:
                        self.history['video'] = self.history['video'][-n_stack:]

                    # self.generate_inputs()q
                    self.execute_action()
                else:
                    print("No frame")
                    is_run = False

            time.sleep(0.5)

        print("Ending run loop")
    def execute_action(self):
        # lift, extension, lateral, roll, gripper
        # action_space = [del_extension, del_height, del_lateral, del_roll, del_gripper]
        # action[0] = np.clip(action[0], -0.01, 0.01)
        
        inputs = self.generate_inputs()
        # with torch.no_grad():
        outputs = self.model(inputs).squeeze() # 11 dimensional
        outputs = torch.nn.functional.softmax(outputs)
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
            'none': 10,
        }
        inv_mapping = {v: k for k, v in mapping.items()}

        action_idx = torch.argmax(outputs).item()
        print("action: ", inv_mapping[action_idx], "output: ", outputs.detach().numpy())
    def generate_inputs(self, save = True):
        
        video = self.history['video'].copy() # subsample n_frames of interest
        n_images = len(video)

        choose_every = n_images // self.n_stack_images

        video = video[::choose_every]
        audio = self.history['audio'].copy()

        if save:
            stacked = np.hstack(video)
            # cv2.imwrite('stacked.jpg', stacked)
            cv2.imshow('stacked', stacked)
            if cv2.waitKey(1) == ord('q'):
                exit()

        # cam_gripper_framestack,audio_clip_g
        # vg_inp: [batch, num_stack, 3, H, W]
        # a_inp: [batch, 1, T]
        
        video = video[-self.n_stack_images:]
        video = [(img).astype(np.float32)/ 255.0 for img in video]
        
        return {
            'video' : video, # list of images
            'audio' : audio, # audio buffer
        }


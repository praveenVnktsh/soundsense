import numpy as np
import torch
import stretch_body.robot
import time
import cv2
import yaml
import rospy
from audio_common_msgs.msg import AudioDataStamped, AudioData
import torchaudio
import soundfile as sf
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String, ByteMultiArray
from models.audio_processor import AudioProcessor
from std_msgs.msg import Header
import json
import os
import datetime
import glob
from pynput import keyboard

keypress = None
def on_press(key):
    global keypress
    keypress = key

def on_release(key):
    global keypress
    keypress = None



class RobotNode:
    def __init__(self, config_path, model_name):
        rospy.init_node("test_model")
        self.model_name = model_name
        self.r = stretch_body.robot.Robot()
        self.boot_robot()
        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        listener.start()
        self.testing = False

        self.model_pub = rospy.Publisher("/load_model", String, queue_size=1)
        for i in range(2):
            self.model_pub.publish(String(data=model_name))
            time.sleep(1)
        
        
        with open(config_path) as info:
            config = yaml.load(info.read(), Loader=yaml.FullLoader)

        config['camera_id'] = '/dev/video6'
        self.image_shape = [config['resized_width_v'], config['resized_height_v']]
        self.hz = config['resample_rate_audio']
        self.audio_n_seconds = config['audio_len']
        cam = config['camera_id']
        self.n_stack_images = config['num_stack']
        print("NUM_STACK", self.n_stack_images)
        self.norm_audio = config['norm_audio']
        self.history = {
            'audio': [],
            'video': [],
            'timestamps': [],
        }
        self.use_audio = "ag" in config['modalities'].split("_")
        if self.use_audio:
            audio_sub = rospy.Subscriber('/audio/audio', AudioData, self.audio_callback)
            # rospy.wait_for_message('/audio/audio', AudioData, timeout=10)
        self.cap  = cv2.VideoCapture(cam)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        

        self.seq_len = config['output_sequence_length']
        self.output_dim = config['action_dim']
        self.temp_ensemble = False

        idx = [0]
        for i in range(1, self.seq_len):
            idx.append(idx[-1] + i)
        self.idx = np.array(idx)

        self.m = 2
        self.weights = np.array([np.exp(-self.m*i) for i in range(self.seq_len)])

        self.temporal_ensemble_arr = np.zeros(((self.seq_len)*(self.seq_len + 1)//2-self.seq_len, self.output_dim))

        self.image_pub = rospy.Publisher('/raw_image', Image, queue_size=1)
        self.audio_pub = rospy.Publisher('/raw_audio', AudioDataStamped, queue_size=1)
        self.images = sorted(glob.glob('/home/hello-robot/soundsense/soundsense/data/sorting/15/video/*.png'))
        # make a uint8 list of actions subscriber
        self.outputs_sub = rospy.Subscriber('/model_outputs', String, self.get_model_inference)
        self.bridge = CvBridge()
        self.last_exec_time = time.time()
        self.idx = 0
    
    def execute_action(self, sequence):
        if abs(self.last_exec_time - time.time()) < 0.1:
            return
        r = self.r
        mapping = {
            'w': 0,
            's': 1,
            'n': 2,
            'm': 3,
            'k': 4,
            'j': 5,
            'l': 6,
            'none': 7
        }
        inv_mapping = {v: k for k, v in mapping.items()}
        big = 0.05
        action_idx = sequence[0]

        movement_resolution = {
            0: big,
            1: -big,
            2: 50,
            3: -50,
            4: -big,
            5: 15 * np.pi/180,
            6: -15 * np.pi/180,
        }
        if inv_mapping[action_idx] in ['w', 's']:
            r.arm.move_by(movement_resolution[action_idx])
        elif inv_mapping[action_idx] in ['a', 'd']:
            r.base.translate_by(movement_resolution[action_idx])
        elif inv_mapping[action_idx] in ['n', 'm']:
            r.end_of_arm.move_by('stretch_gripper', movement_resolution[action_idx])
        elif inv_mapping[action_idx] in ['j', 'l']:
            r.end_of_arm.move_by('wrist_roll', movement_resolution[action_idx])
        elif inv_mapping[action_idx] in ['i', 'k']:
            if inv_mapping[action_idx] == 'i':
                r.end_of_arm.move_by('wrist_pitch', np.pi * 6/ 180.0)
            else:
                r.end_of_arm.move_by('wrist_pitch', -np.pi * 6/ 180.0)
        if inv_mapping[action_idx] != 'none' and not self.testing:
            r.push_command()
        self.last_exec_time = time.time()
        


    def get_model_inference(self, data):
        predictions = np.array(json.loads(data.data))
        # self.execute_action(predictions)
        

    def audio_callback(self, data):
        audio = np.frombuffer(data.data, dtype=np.int16).tolist()
        self.history['audio']+= (audio)
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

        self.r.arm.move_to(0.1)
        self.r.lift.move_to(1.075)
        self.r.end_of_arm.move_to('wrist_pitch', 0.0)
        self.r.end_of_arm.move_to('wrist_yaw', 0.0)
        self.r.end_of_arm.move_to('stretch_gripper', 100)
        self.r.end_of_arm.move_to('wrist_roll', 0.0)
        self.r.head.move_to('head_pan', -np.pi/2)
        self.r.head.move_to('head_tilt', -np.pi/6)
        self.r.push_command()
        time.sleep(5)
        print("Robot ready to run model.")
    
    def get_image(self):
        h, w = self.image_shape 
        ret, frame = self.cap.read()
        # frame = np.zeros((h, w, 3), dtype = np.uint8)
        # ret = True
        # self.idx += 1
        # self.idx %= len(self.images)
        # frame = cv2.imread(self.images[self.idx])
        # ret = True
        # time.sleep(0.1)
        frame = cv2.resize(frame, (h, w), interpolation=cv2.INTER_NEAREST)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not ret:
            return None
        return frame

    def run_loop(self, visualize = False):
        is_run = True
        
        loop_start_time = time.time()
        n_stack = self.n_stack_images * self.audio_n_seconds
        last_exec_time = time.time()
        while is_run:
            
            frame = self.get_image()
            curtime = time.time()
            self.history['video'].append(frame)
            self.history['timestamps'].append(curtime)
            if len(self.history['video']) >= n_stack:
                while self.history['timestamps'][-1] - self.history['timestamps'][0] > self.audio_n_seconds:
                    self.history['video'] = self.history['video'][1:]
                    self.history['timestamps'] = self.history['timestamps'][1:]
            
            if time.time() - loop_start_time > self.audio_n_seconds: # warmup
                if abs(last_exec_time - time.time() )> 0.1:
                    # print(self.history['timestamps'][-1] - time.time(), self.history['timestamps'][0] - time.time())
                    self.generate_inputs()
                    last_exec_time = time.time()
            

            # print(len(self.history['video']), len(self.history['audio']))

        print("Ending run loop")

    def generate_inputs(self, save = True):
        
        video = self.history['video'].copy() # subsample n_frame s of interest
        n_images = len(video)

        video = video[(n_images % self.n_stack_images):]
        choose_every = n_images // self.n_stack_images
        video = video[::choose_every]
        assert len(video) == self.n_stack_images
        
        if self.use_audio:
            audio_to_send = np.array(self.history['audio'], dtype = np.int16).tobytes()
        
        else:
            audio_to_send = []

        stacked = np.hstack(video)
        print(np.mean(stacked), np.std(stacked))
        c = str(keypress).replace("'", '')
        # print("Keypress: ", c)

        header = Header(
            stamp = rospy.Time.now(),
            frame_id = c
        )
        imgmsg = self.bridge.cv2_to_imgmsg(stacked, encoding = 'rgb8')
        # imgmsg = Image()
        # imgmsg.data = stacked.tostring()
        # imgmsg.height = stacked.shape[0]
        # imgmsg.width = stacked.shape[1]
        # imgmsg.encoding = 'rgb8'
        imgmsg.header = header
        self.image_pub.publish(imgmsg)
        header.frame_id = self.model_name
        # sf.write('audio.wav', np.array(self.history['audio'], dtype = np.int16), 16000)
        self.audio_pub.publish(AudioDataStamped(
            audio = AudioData(data = audio_to_send),
            header = header
        ))
        self.model_pub.publish(String(data=self.model_name))

        if c == 'q':
            exit()



    

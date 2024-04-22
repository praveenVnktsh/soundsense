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
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class RobotNode:
    def __init__(self, config_path, model, testing= False):
        self.r = stretch_body.robot.Robot()
        self.boot_robot()

        self.testing = False
        
        with open(config_path) as info:
            params = yaml.load(info.read(), Loader=yaml.FullLoader)

        
        
        params['camera_id'] = '/dev/video6'

        self.image_shape = [params['resized_width_v'], params['resized_height_v']]

        self.hz = params['resample_rate_audio']
        self.audio_n_seconds = params['audio_len']
        cam = params['camera_id']
        self.n_stack_images = params['num_stack']
        self.norm_audio = params['norm_audio']
        self.history = {
            'audio': [],
            'video': [],
        }
        self.use_audio = "ag" in params['modalities'].split("_")
        if self.use_audio:
            audio_sub = rospy.Subscriber('/audio/audio', AudioData, self.callback)
            print("Waiting for audio data...")
            rospy.wait_for_message('/audio/audio', AudioData, timeout=10)
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.hz, 
                n_fft=512, 
                hop_length=int(self.hz * 0.01), 
                n_mels=64
            )
        self.cap  = cv2.VideoCapture(cam)
        self.model = model

        self.seq_len = 10
        self.output_dim = 11
        self.temp_ensemble = True

        idx = [0]
        for i in range(1, self.seq_len):
            idx.append(idx[-1] + i)
        self.idx = np.array(idx)

        self.m = 2
        self.weights = np.array([np.exp(-self.m*i) for i in range(self.seq_len)])

        self.temporal_ensemble_arr = np.zeros(((self.seq_len)*(self.seq_len + 1)//2-self.seq_len, self.output_dim))

        self.inputs_pub = rospy.Publisher('/model_inputs', CompressedImage, queue_size=10)
        self.inputs_sub = rospy.Subscriber('/model_inputs', CompressedImage, self.subscribe_inputs)
        self.bridge = CvBridge()

    
    def callback(self, data):
        audio = np.frombuffer(data.data, dtype=np.int16).copy().astype(np.float64)
        audio /= 32768
        audio = audio.tolist()
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

        r.lift.move_to(0.9)
        r.arm.move_to(0.2)
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
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (h, w))
        # if self.bgr_to_rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not ret:
            return None
        return frame

    def run_loop(self, visualize = False):
        is_run = True
        start_time = time.time()
        loop_rate = 10
        loop_start_time = time.time()
        
        n_stack = self.n_stack_images * self.audio_n_seconds
        hist_actions = []
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

                    # try:
                    #     hist_actions = self.execute_action(hist_actions)
                    #     if len(hist_actions)>0:
                    #         np.savetxt("actions.txt", hist_actions,fmt='%s')
                    # except:
                    #     self.boot_robot()
                    #     is_run = False
                else:
                    print("No frame")
                    is_run = False

        print("Ending run loop")
    def clip_resample(self, audio, audio_start, audio_end):
        left_pad, right_pad = torch.Tensor([]), torch.Tensor([])
        # print("au_size", audio.size())
        if audio_start < 0:
            left_pad = torch.zeros((audio.shape[0], -audio_start))
            audio_start = 0
        if audio_end >= audio.size(-1):
            right_pad = torch.zeros((audio.shape[0], audio_end - audio.size(-1)))
            audio_end = audio.size(-1)
        audio_clip = torch.cat(
            [left_pad, audio[:, audio_start:audio_end], right_pad], dim=1
        )
        
        # audio_clip = torchaudio.functional.resample(audio_clip, self.hz, self.resample_rate_audio)
        # print("Inside clip_resample output shape", audio_clip.shape)
        
        return audio_clip
    def generate_inputs(self, save = True):
        
        video = self.history['video'].copy() # subsample n_frame s of interest
        n_images = len(video)

        choose_every = n_images // self.n_stack_images

        video = video[::choose_every]
        
        if self.use_audio:
            starttime = time.time()
            audio = torch.tensor(self.history['audio']).float()
            audio = audio.unsqueeze(0).unsqueeze(0)
            mel = self.mel(audio)
            mel = np.log(mel + 1)
            if self.norm_audio:
                mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
                mel -= mel.mean()            # TODO why are we doing this after the previous step?
            print("AUDIO_MEL", mel.min(), mel.max())
            print("RAW_AUDIO", audio.min(), audio.max())
            print("Time to process audio: ", time.time() - starttime)
        else:
            mel = None

        if save:
            stacked = np.hstack(video)
            # cv2.imwrite('stacked.jpg', stacked)
        
            stacked = cv2.resize(stacked, (0, 0), fx = 2, fy = 2)
            
            if self.use_audio:
                import matplotlib.pyplot as plt
                # plt.ion()
                # plt.imshow(mel.squeeze().numpy(), cmap = 'viridis',origin='lower',)
                # plt.show()
                temp =  mel.clone().numpy().squeeze()
                temp -= temp.min()
                temp /= temp.max()
                temp *= 255
                temp = temp.astype(np.uint8)
                temp = cv2.resize(temp, stacked.shape[:2][::-1])
                stacked = np.vstack([stacked, temp])
                
            
            cv2.imshow('stacked', stacked)
            cv2.waitKey(1)
                
            
            

        # cam_gripper_framestack,audio_clip_g
        # vg_inp: [batch, num_stack, 3, H, W]
        # a_inp: [batch, 1, T]
        
        video = video[-self.n_stack_images:]
        video = [(img).astype(float)/ 255 for img in video]
        
        return {
            'video' : video, # list of images
            'audio' : mel, # audio buffer
        }


    def publish_inputs(self, inputs):
        images = []
        for img in inputs['video']:
            img *= 255
            img = img.astype(np.uint8)
            img = cv2.resize(img, (256, 256))
            images.append(img)
        if inputs['audio'] is not None:
            audio = inputs['audio'].clone().numpy().squeeze()
            audio *= 255
            audio = audio.astype(np.uint8)
            audio = cv2.resize(audio, (256, 256))
            images.append(audio)

        compressed_images = []
        for img in images:
            compressed_image_msg = self.bridge.cv2_to_compressed_imgmsg(img)
            compressed_images.append(compressed_image_msg)

        compressed_images_msg = CompressedImage()
        compressed_images_msg.format = "jpeg"
        compressed_images_msg.data = np.array([compressed_image_msg.data for compressed_image_msg in compressed_images])
        
        self.inputs_pub.publish(compressed_images_msg)


    def subscribe_inputs(self, data):
        try:
            # Convert the compressed image data to a numpy array
            data = np.frombuffer(data.data, np.uint8)
            # Decode the compressed images
            images = []
            idx = 0
            while idx < len(data):
                # Read the length of the next compressed image
                length = int.from_bytes(data[idx:idx+4], byteorder='little')
                idx += 4
                # Extract the compressed image data
                img_data = data[idx:idx+length]
                idx += length
                # Decompress the image
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                images.append(img)
            
            # Process the images (e.g., display or save them)
            for i, img in enumerate(images):
                cv2.imshow("Image {}".format(i), img)
            cv2.waitKey(1000)  # Adjust the delay as needed

            inputs = {
                'audio': [],
                'video': [],
            } 

            for i in range(self.num_stack):
                img = images[i]
                img = cv2.resize(img, (self.image_shape[0], self.image_shape[1]))
                img = img.astype(np.float32) / 255
                inputs['video'].append(img)
            if self.use_audio:
                audio = images[self.num_stack]
                audio = cv2.resize(audio, (self.audio_shape[0], self.audio_shape[1]))  # TODO audio shape not defined
                audio = audio.astype(np.float32) / 255
                inputs['audio'] = audio

            # Call execute action with the inputs

        except Exception as e:
            rospy.logerr("Error decoding images: %s", e)


    def execute_action(self, hist_actions=[]):
        # lift, extension, lateral, roll, gripper
        # action_space = [del_extension, del_height, del_lateral, del_roll, del_gripper]
        # action[0] = np.clip(action[0], -0.01, 0.01)
        r = self.r
        inputs = self.generate_inputs()
        starttime = time.time()
        outputs = self.model(inputs) # 11 dimensional
        print("Time to run model: ", time.time() - starttime)
        outputs_model = outputs.clone()
        if self.temp_ensemble:
            outputs = outputs.squeeze(0).detach().numpy()
            # print("tea shape",self.temporal_ensemble_arr.shape)
            # print("outputs shape", outputs.shape)
            self.temporal_ensemble_arr = np.concatenate([self.temporal_ensemble_arr, outputs], axis = 0)
            
            # print("TE SHAPE: ", self.temporal_ensemble_arr.shape)
            values = self.temporal_ensemble_arr[self.idx, :]
            outputs = np.average(values, weights=self.weights, axis = 0)
            # print("OUTPUTS: ", outputs)
            # print(outputs.shape)
            outputs_softmax = torch.nn.functional.softmax(torch.tensor(outputs))
            self.temporal_ensemble_arr = np.delete(self.temporal_ensemble_arr, self.idx, axis=0)
            # print("TE SHAPE after delete: ", self.temporal_ensemble_arr.shape)
        else:
            outputs_softmax = torch.nn.functional.softmax(outputs.squeeze(0)[0])
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
            'none': 10,
        }
        inv_mapping = {v: k for k, v in mapping.items()}

        action_idx = torch.argmax(outputs_softmax).item()
        action_model_idx = torch.argmax(torch.nn.functional.softmax(outputs_model.squeeze(0)[0])).item()
        print("action te: ", inv_mapping[action_idx], "action:",inv_mapping[action_model_idx], "output: ", outputs)
        hist_actions.append((inv_mapping[action_idx].ljust(6), inv_mapping[action_model_idx].ljust(6)))
        big = 0.05
        movement_resolution = {
            0: big,
            1: -big,
            2: big,
            3: -big,
            4: 50,
            5: -50,
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
        elif action_idx in [8, 9]:
            r.end_of_arm.move_by('wrist_roll', movement_resolution[action_idx])
        elif action_idx in [6, 7]:
            r.lift.move_by(movement_resolution[action_idx])
        if action_idx != 10 and not self.testing:
            r.push_command()

        return hist_actions

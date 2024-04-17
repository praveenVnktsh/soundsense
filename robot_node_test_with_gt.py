import os
import numpy as np
import torch
# import stretch_body.robot
import time
import cv2
import yaml
# import rospy
# from audio_common_msgs.msg import AudioDataStamped, AudioData
import glob
import soundfile as sf
import torchaudio

class RobotNode:
    def __init__(self, config_path, model, is_unimodal = False, testing = False, run_id='20'):
        # self.r = stretch_body.robot.Robot()
        # self.boot_robot()
        
        with open(config_path) as info:
            params = yaml.load(info.read(), Loader=yaml.FullLoader)

        self.image_shape = [params['resized_height_v'], params['resized_width_v']]

        self.hz = 16000
        self.audio_n_seconds = params['audio_len']
        self.use_audio = "ag" in params['modalities'].split("_")
        self.norm_audio = params['norm_audio']
        print("Using audio:", self.use_audio)

        self.n_stack_images = params['num_stack']
        self.history = {
            'audio': [0] * self.hz * self.audio_n_seconds,
            'video': [],
        }
        self.run_id = run_id
        os.makedirs(f'/home/punygod_admin/SoundSense/soundsense/gt_inference/{run_id}', exist_ok = True)
        os.makedirs(f'/home/punygod_admin/SoundSense/soundsense/gt_inference/{run_id}/video', exist_ok = True)
        if not is_unimodal:
            data = sf.read(f'/home/punygod_admin/SoundSense/soundsense/data/mulsa/data/{run_id}/processed_audio.wav')[0]
            # data = sf.read(f'/home/hello-robot/soundsense/soundsense/stretch/data/data_two_cups/{run_id}/processed_audio.wav')[0]
            self.total_audio = torchaudio.functional.resample(torch.tensor(data), 48000, 16000).numpy()
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.hz, 
                n_fft=512, 
                hop_length=int(self.hz * 0.01), 
                n_mels=64
            )
        self.idx = 0
        self.model = model
        self.images = sorted(glob.glob(f'/home/punygod_admin/SoundSense/soundsense/data/mulsa/data/{run_id}/video/*.png'))
        # self.images = sorted(glob.glob(f'/home/hello-robot/soundsense/soundsense/stretch/data/data_two_cups/{run_id}/video/*.png'))
    
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
        
        audio_clip = torchaudio.functional.resample(audio_clip, self.hz, 16000)
        return audio_clip
    
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
        if self.idx >= len(self.images):
            return None
        print("idx", self.idx)
        path = self.images[self.idx]
        frame = cv2.imread(path)
        frame = cv2.resize(frame, (h, w))
        # frame = np.random.randint(0, 255, (h, w, 3)).astype(np.uint8)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ## same frames
        cv2.imwrite(f'/home/punygod_admin/SoundSense/soundsense/gt_inference/{self.run_id}/video/frame_{self.idx}.jpg', frame)
        return frame

    def run_loop(self, visualize = False):
        is_run = True
        start_time = time.time()
        loop_rate = 10
        
        
        n_stack = self.n_stack_images * self.audio_n_seconds

        # warmup first 3 seconds
        
        # for i in range(300):
        #     frame = self.get_image()
        #     self.history['video'].append(frame)

        audio_per_frame = self.hz / loop_rate
        hist_actions = []
        while is_run:
            frame = self.get_image()
            self.history['video'].append(frame)
            if frame is not None or self.idx % 5 == 0:
                if len(self.history['video']) < n_stack:
                    continue
                if len(self.history['video']) > n_stack:
                    self.history['video'] = self.history['video'][-n_stack:]

                audio_end = int(self.idx * self.hz / loop_rate )
                audio_start = audio_end - self.hz * self.audio_n_seconds
                
                print("audio_start", audio_start/self.hz, "audio_end", audio_end/self.hz, 'completion', audio_end/len(self.total_audio))
                print("Frame time:", self.idx / loop_rate)
                if audio_start < 0:
                    pad_length = -audio_start
                    self.history['audio'] = np.array([0] * pad_length + self.total_audio[:audio_end].tolist())
                else:
                    self.history['audio'] = self.total_audio[audio_start:audio_end]
                try:
                    hist_actions = self.execute_action(hist_actions)
                except Exception as e:
                    print("Error in execute action", e)
                    is_run = False
                # time.sleep(0.1)
        filepath = f'/home/punygod_admin/SoundSense/soundsense/gt_inference/{self.run_id}/actions.txt'
        np.savetxt(filepath, hist_actions, fmt='%s')
        print("Ending run loop")

    def execute_action(self, hist_actions=[]):
        # lift, extension, lateral, roll, gripper
        # action_space = [del_extension, del_height, del_lateral, del_roll, del_gripper]
        # action[0] = np.clip(action[0], -0.01, 0.01)
        
        inputs = self.generate_inputs(False)
        # with torch.no_grad():
        print("inputs", inputs['video'][0].shape)
        outputs = self.model(inputs).squeeze() # 11 dimensional
        print("model output shape", outputs.shape)
        outputs = outputs[0] # take 0th action from sequence predicted
        outputs = torch.nn.functional.softmax(outputs, dim = 0)
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
        print("action: ", inv_mapping[action_idx], "output: ", outputs.cpu().detach().numpy().tolist())
        hist_actions.append(inv_mapping[action_idx])
        return hist_actions
    def generate_inputs(self, save = True):
        print("video shape", len(self.history['video']))
        video = self.history['video'].copy() # subsample n_frames of interest
        n_images = len(video)
        ## This is different then training?
        choose_every = n_images // self.n_stack_images

        video = video[::choose_every]
        mel = None
        if self.use_audio:
            audio = self.history['audio'].copy()
            audio = torch.tensor(audio).float()
            audio = audio.unsqueeze(0).unsqueeze(0)
            mel = self.mel(audio)
            eps = 1e-8
            mel = np.log(mel + eps)
            
            print((audio).numel() / self.hz)
            print(audio.max(), audio.min(), audio.mean(), mel.max(), mel.min(), mel.mean())
            if self.norm_audio:
                mel /= mel.sum(dim = -2, keepdim = True)

        if save:
            stacked = np.hstack(video)
            # cv2.imwrite('stacked.jpg', stacked)
            stacked = cv2.resize(stacked,(0, 0), fx = 3, fy = 3)
            cv2.imshow('stacked', stacked)
            if cv2.waitKey(1) == ord('q'):
                exit()

            if self.use_audio:
                import matplotlib.pyplot as plt
                temp = mel.squeeze().numpy()
                temp -= -15
                temp /= 30
                temp *= 255
                temp = temp.astype(np.uint8)
                temp = cv2.resize(temp, (0, 0), fx = 3, fy = 3)
                temp = cv2.flip(temp, 0)
                plt.ion()
                plt.cla()
                t = np.linspace(0, (audio.numel()) / self.hz, audio.numel())
                plt.plot(t, audio.numpy().squeeze(), color='b')
                plt.ylim(-1, 1)
                plt.show()

                cv2.imshow('mel', temp)
                if cv2.waitKey(1) == ord('q'):
                    exit()
                # plt.ion()
                # plt.imshow(mel.squeeze().numpy(), cmap = 'viridis',origin='lower',)
                # plt.show()

        # cam_gripper_framestack,audio_clip_g
        # vg_inp: [batch, num_stack, 3, H, W]
        # a_inp: [batch, 1, T]
        print("Old video shape", video[0].shape, len(video))

        video = video[-self.n_stack_images:]
        # video = [(img)/ 255.0 for img in video]
        # video = [(img).astype(np.float32)/ 255.0 for img in video]
        print("video shape", type(video[0]), video[0].shape, len(video))
        return {
            'video' : video, # list of images
            'audio' : mel, # audio buffer
        }


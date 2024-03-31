import os
import torch
import yaml
import torchvision.transforms as T

import numpy as np
from PIL import Image
import random
from torch.utils.data.dataset import Dataset
import torchaudio
import soundfile as sf
import json
import glob
import matplotlib.pyplot as plt

class ImitationEpisode(Dataset):
    def __init__(self, 
            config,
            run_id, 
            train=True):
        # print("d_idx", dataset_idx)
        super().__init__()
        self.train = train
        
        super().__init__()
        # self.logs = pd.read_csv(log_file)
        self.run_id = run_id
        self.sample_rate_audio = 48000 ##TODO
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate_audio,
            n_fft=int(self.sample_rate_audio * 0.025),
            hop_length=int(self.sample_rate_audio * 0.01),
            n_mels=64,
            center=False,
        )
        
        self.fps = config["fps"]
        self.audio_len = config['audio_len']
        self.sample_rate_audio = config["sample_rate_audio"]
        self.resample_rate_audio = config['resample_rate_audio']
        self.modalities = config['modalities'].split("_")
        self.resized_height_v = config['resized_height_v']
        self.resized_width_v = config['resized_width_v']
        self.nocrop = not config['is_crop']
        self.crop_percent = config['crop_percent']

        self.dataset_root = config['dataset_root']
        
        # Number of images to stack
        self.num_stack = config['num_stack']
        # Number of frames to skip
        self.frameskip = self.fps * self.audio_len // self.num_stack
        # Maximum length of images to consider for stacking
        self.max_len = (self.num_stack - 1) * self.frameskip
        # Number of audio samples for one image idx
        self.resolution = self.sample_rate_audio // self.fps  
        
        # augmentation parameters
        self._crop_height_v = int(self.resized_height_v * (1.0 - self.crop_percent))
        self._crop_width_v = int(self.resized_width_v * (1.0 - self.crop_percent))

        self.actions, self.audio_gripper, self.episode_length, self.image_paths = self.get_episode()

        # self.action_dim = config['action_dim']
        
        if self.train:
            self.transform_cam = T.Compose(
                [
                    T.Resize((self.resized_height_v, self.resized_width_v)),
                    T.ColorJitter(brightness=0.1, contrast=0.02, saturation=0.02),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        else:
            self.transform_cam = T.Compose(
                [
                    T.Resize((self.resized_height_v, self.resized_width_v)),
                    # T.CenterCrop((self._crop_height_v, self._crop_width_v)),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

    def load_image(self, idx):
        img_path = self.image_paths[idx]
        image = (
            torch.as_tensor(np.array(Image.open(img_path))).float().permute(2, 0, 1)
            / 255
        )
        # normalization
        # image -= 0.5
        # image /= 0.5
        

        return image
    
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
        # print(f"start {audio_start}, end {audio_end} left {left_pad.size()}, right {right_pad.size()}. audio_clip {audio_clip.size()}")
        audio_clip = torchaudio.functional.resample(audio_clip, self.sample_rate_audio, self.resample_rate_audio)
        # print("Inside clip_resample output shape", audio_clip.shape)
        
        return audio_clip

    def __len__(self):
        return self.episode_length

    def get_episode(self):            
        episode_folder = os.path.join(self.dataset_root, self.run_id)

        with open(os.path.join(episode_folder, "actions.json")) as ts:
            actions = json.load(ts)

        if "ag" in self.modalities:
            if os.path.exists(os.path.join(episode_folder, "processed_audio.wav")):
                audio_gripper1 = sf.read(os.path.join(episode_folder, "processed_audio.wav"))[0]
            else:
                audio_gripper1 = None
            
            audio_gripper = [
                x for x in audio_gripper1 if x is not None
            ]
            audio_gripper = torch.as_tensor(np.stack(audio_gripper, 0))
            audio_gripper = (audio_gripper).reshape(1,-1)
        else:
            audio_gripper = None

        image_paths = sorted(glob.glob(f'{self.dataset_root}/{self.run_id}/video/*.png'))

        return (
            actions,
            audio_gripper,
            min(len(actions), len(image_paths)),
            image_paths
        )

    def __len__(self):
        return self.episode_length

    def __getitem__(self, idx):
        start_idx = idx - self.max_len

        # Frames to stack
        frame_idx = np.arange(start_idx, idx + 1, self.frameskip)
        frame_idx[frame_idx < 0] = 0
        frame_idx[frame_idx >= self.episode_length] = self.episode_length - 1
    
        if "vg" in self.modalities:
            # stacks from oldest to newest left to right!!!!
            cam_gripper_framestack = torch.stack(
                [
                    self.transform_cam(
                        self.load_image(idx)
                    )
                    for idx in frame_idx
                ],
                dim=0,
            )
            # if idx > 500:
            #     stacked = [img.permute(1, 2, 0).numpy()*0.5 + 0.5 for img in cam_gripper_framestack]
            #     stacked = np.hstack(stacked)
            #     plt.imsave('image.png',stacked) 
            #     for idx in frame_idx:
            #         print(self.actions[idx])
            #     exit()
        else:
            cam_gripper_framestack = None

        # Random cropping
        if self.train:
            img = self.transform_cam(
                self.load_image(idx)
            )
            if not self.nocrop:
                i_v, j_v, h_v, w_v = T.RandomCrop.get_params(
                    img, output_size=(self._crop_height_v, self._crop_width_v)
                )
            else:
                i_v, h_v = (
                    self.resized_height_v - self._crop_height_v
                ) // 2, self._crop_height_v
                j_v, w_v = (
                    self.resized_width_v - self._crop_width_v
                ) // 2, self._crop_width_v

            if "vg" in self.modalities:
                cam_gripper_framestack = cam_gripper_framestack[
                    ..., i_v : i_v + h_v, j_v : j_v + w_v
                ]

        audio_end = idx * self.resolution
        audio_start = audio_end - self.audio_len * self.sample_rate_audio
        
        if self.audio_gripper is not None:
            audio_clip_g = self.clip_resample(
                self.audio_gripper, audio_start, audio_end
            ).float()
        else:
            audio_clip_g = 0
 
        xyzgt = torch.Tensor(self.actions[idx])

        return (
            (cam_gripper_framestack,
            audio_clip_g,),
            xyzgt,
        )
import os
import torch
import yaml
import sys
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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoProcessor
import transformers
import cv2
sys.path.append('/home/punygod_admin/SoundSense/soundsense/models')
from audio_processor import AudioProcessor



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
        self.fps = config["fps"]
        self.audio_len = config['audio_len']
        self.sample_rate_audio = config["sample_rate_audio"]
        self.resample_rate_audio = config['resample_rate_audio']
        self.modalities = config['modalities'].split("_")
        self.resized_height_v = config['resized_height_v']
        self.resized_width_v = config['resized_width_v']
        # self.nocrop = not config['is_crop']
        # self.crop_percent = config['crop_percent']
        self.action_dim = config['action_dim']
        
        # self.input_past_actions = config['input_past_actions']
        # self.stack_past_actions = config['stack_past_actions']
        # self.stack_future_actions_dim = config['stack_future_actions_dim']
        # self.output_model = config['output_model']
        self.output_sequence_length = config['output_sequence_length']
        self.action_history_length = config['action_history_length']
        self.dataset_root = config['dataset_root']
        self.norm_audio = config['norm_audio']
        
        # Number of images to stack
        self.num_stack = config['num_stack']
        # Number of frames to skip
        self.frameskip = self.fps * self.audio_len // self.num_stack + 1
        # Maximum length of images to consider for stacking
        self.max_len = (self.num_stack - 1) * self.frameskip
        # Number of audio samples for one image idx
        
        # augmentation parameters
        # self._crop_height_v = int(self.resized_height_v * (1.0 - self.crop_percent))
        # self._crop_width_v = int(self.resized_width_v * (1.0 - self.crop_percent))

        self.actions, self.audio_gripper, self.episode_length, self.image_paths = self.get_episode()
        if self.audio_gripper is not None:
            self.resolution = (self.audio_gripper.numel()) // self.episode_length
            self.audio_processor = AudioProcessor(config)
        # self.action_dim = config['action_dim']
        
        if self.train:
            p_apply = np.array([
                1, 1, 1, 0.1
            ], dtype=np.float32)
            p_apply /= p_apply.sum()
            p_apply *= 0.5
            self.transform_cam = A.Compose([
                # A.Resize(height=self.resized_height_v, width=self.resized_width_v),
                A.GaussianBlur(
                    sigma_limit=(0.2, 0.6),
                    p=p_apply[0]
                ),  # Gaussian blur with 10% probability
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15,
                        contrast_limit=0.15,
                        p = 0.5,
                    ),# Random brightness and contrast adjustments with 20% probability
                    A.ColorJitter(
                        brightness=0, contrast=0, saturation=0, hue=0.1,
                        p = 0.5
                    ),
                ], p=p_apply[1]),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=15, 
                    p=p_apply[2]
                ),
                A.ZoomBlur(
                    max_factor=1.11,
                    p = p_apply[3]
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], max_pixel_value= 1.0),
                ToTensorV2(),
            ], additional_targets= {
                f'image{i}': 'image' for i in range(self.num_stack)})
        else:
            self.transform_cam = A.Compose([
                # A.Resize(height=self.resized_height_v, width=self.resized_width_v),
                
                A.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], max_pixel_value= 1.0),
                ToTensorV2(),
            ])

        
        transformers.utils.logging.set_verbosity_error()
        

    def load_image(self, idx):
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path)).astype(np.float32) / 255.0 # RGB FORMAT ONLY
        return image
    
    def __len__(self):
        return self.episode_length

    def get_episode(self):            
        episode_folder = os.path.join(self.dataset_root, self.run_id)

        with open(os.path.join(episode_folder, "actions.json")) as ts:
            actions = json.load(ts)

        if "ag" in self.modalities:
            print("Loading audio for", episode_folder[-5:])
            if os.path.exists(os.path.join(episode_folder, "processed_audio.wav")):
                audio_gripper1 = sf.read(os.path.join(episode_folder, "processed_audio.wav"))[0]
                # print("Audio loaded")
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
        if len(actions) != len(image_paths):
            print("Mismatch", len(actions) - len(image_paths), self.run_id)
            exit()
            
        actions = torch.Tensor(actions)
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
            if self.train:
                images = {f'image{i}' : self.load_image(fidx) for i, fidx in enumerate(frame_idx)}
                images['image'] = images['image0']
                transformed = self.transform_cam(**images)
                cam_gripper_framestack = torch.stack(
                    [
                        transformed[f'image{i}']
                        for i in range(len(frame_idx))
                    ],
                    dim=0,
                )
            else:
                cam_gripper_framestack = torch.stack(
                    [
                        self.transform_cam(
                            image = self.load_image(fidx)
                        )['image']
                        for fidx in frame_idx
                    ],
                    dim=0,
                )         
        else:
            cam_gripper_framestack = 0

        if self.audio_gripper is not None:
            audio_end = idx * self.resolution
            audio_start = audio_end - self.audio_len * self.resample_rate_audio
            mel = self.audio_processor.process(self.audio_gripper, audio_start, audio_end, clip_and_resample=True).float()
        else:
            mel = 0
 
        start_idx = idx
        end_idx = idx + self.output_sequence_length
        padding = torch.tensor([]) if end_idx <= self.episode_length else torch.tensor([[0] * (self.action_dim - 1) + [1]] * (end_idx - self.episode_length))

        action_trajectory = torch.cat(
            [
                self.actions[start_idx:end_idx],
                padding
            ],
            dim=0
        )

        start_idx = idx - self.action_history_length
        end_idx = idx 
        padding = torch.tensor([]) if start_idx >= 0 else torch.tensor([[0] * (self.action_dim - 1) + [1]] * (-start_idx))

        action_history = torch.cat(
            [
                padding,
                self.actions[start_idx:end_idx]
            ],
            dim=0
        )

        

        # print(cam_gripper_framestack.shape)
        return (
            (cam_gripper_framestack,
            mel),
            (action_trajectory, action_history),
        )
    
if __name__ == "__main__":
    import os
    os.makedirs('temp', exist_ok=True)
    dataset = ImitationEpisode(
        config = {
            'fps': 10,
            'audio_len': 3,
            'sample_rate_audio': 16000,
            'resample_rate_audio': 16000,
            'modalities': 'vg_ag',
            'resized_height_v': 75,
            'resized_width_v': 100,
            'action_dim' : 11,
            'is_crop': False,
            'crop_percent': 0.1,
            'output_model' : 'layered',
            'stack_past_actions': False,
            'stack_future_actions': False,
            'stack_future_actions_dim': 6,
            'input_past_actions': False,
            'input_past_actions_dim': 6,
            'history_encoder_dim': 32,
            'output_sequence_length' : 1,
            'action_history_length': 0,
            'dataset_root': '/home/punygod_admin/SoundSense/soundsense/data/mulsa/sorting',
            # 'dataset_root': '/home/praveen/dev/mmml/soundsense/data/',
            'num_stack': 6,
            'norm_audio' : True
        },
        run_id = "20",
        train=True
    )
    print("Dataset size", len(dataset))
    i = 18
    print("Index", i)
    (cam_gripper_framestack, mel_spec), (xyzgt, frame_idx) = dataset[i]
    # save images
    stacked = []
    print("actions", xyzgt.shape)
    print("melspec", mel_spec.shape, "min", mel_spec.min(), "max", mel_spec.max())
    print("framestack.shape", cam_gripper_framestack.shape)
    for idx, img in enumerate(cam_gripper_framestack):
        print("image ", idx, "min", img.min(), "max", img.max())
        img = img.permute(1, 2, 0).numpy() * 0.28 + 0.48
        img = np.clip(img, 0, 1)
        stacked.append(img.copy())
    stacked = np.hstack(stacked)
    mel = mel_spec[0].numpy()
    
    mel = cv2.resize(mel, stacked.shape[:2][::-1])
    mel -= mel.min()
    mel /= mel.max()
    mel = cv2.applyColorMap((mel * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    stacked = (stacked * 255).astype(np.uint8)    
    viz = np.vstack([stacked, mel])
    # plt.imsave('temp/viz.png', viz)
    cv2.imwrite('temp/viz.png', viz)
    # plt.imsave('temp/melspec.png', mel_spec[0].numpy(), cmap='viridis', origin='lower', )
    # plt.imsave(f'temp/0.png', stacked)

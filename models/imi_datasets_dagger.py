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
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
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
        self.episode_folder = os.path.join(self.dataset_root, self.run_id)
        print(self.episode_folder)
        self.actions, self.audio_paths, self.episode_length, self.image_paths = self.get_episode()

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
                
                A.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], max_pixel_value= 1.0),
                ToTensorV2(),
            ])
        else:
            self.transform_cam = A.Compose([
                # A.Resize(height=self.resized_height_v, width=self.resized_width_v),
                
                A.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], max_pixel_value= 1.0),
                ToTensorV2(),
            ])
        

    def load_image(self, idx):
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path)).astype(np.float32) / 255.0 # RGB FORMAT ONLY
        w = 100
        images = [
            image[:, i*w : (i+1) * w] for i in range(self.num_stack)
        ]
        # for im in images:
        #     print(im.shape, image.shape)
        return images
    
    def __len__(self):
        return self.episode_length

    def get_episode(self):            

        with open(os.path.join(self.episode_folder, "actions.json")) as ts:
            actions = json.load(ts)

        
        image_paths = sorted(glob.glob(f'{self.dataset_root}/{self.run_id}/video/*.png'))
        audio_paths = sorted(glob.glob(f'{self.dataset_root}/{self.run_id}/audio/*.npy'))
        if len(actions) != len(image_paths):
            print("Mismatch", len(actions) - len(image_paths), self.run_id)
            exit()
        actions = torch.Tensor(actions)
        return (
            actions,
            audio_paths,
            min(len(actions), len(image_paths)),
            image_paths
        )

    def __len__(self):
        return self.episode_length


    def __getitem__(self, idx):

        if "vg" in self.modalities:
            # stacks from oldest to newest left to right!!!!
            images = self.load_image(idx)
            cam_gripper_framestack = torch.stack(
                [
                    self.transform_cam(image=image)["image"]
                    for image in images
                ]
            )

        else:
            cam_gripper_framestack = 0

        mel = torch.tensor(np.load(self.audio_paths[idx])).float().squeeze(0)
 
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
            'dataset_root': '/home/punygod_admin/SoundSense/soundsense/data/mulsa/dagger_1',
            # 'dataset_root': '/home/praveen/dev/mmml/soundsense/data/',
            'num_stack': 6,
            'norm_audio' : True
        },
        run_id = "1",
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
    mel = mel_spec.numpy().squeeze()
    
    mel = cv2.resize(mel, stacked.shape[:2][::-1])
    mel -= mel.min()
    mel /= mel.max()
    mel = cv2.applyColorMap((mel * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    stacked = (stacked * 255).astype(np.uint8)    
    viz = np.vstack([stacked, mel])
    print("Saving")
    # plt.imsave('temp/viz.png', viz)
    cv2.imwrite('temp/viz.png', viz)
    # plt.imsave('temp/melspec.png', mel_spec[0].numpy(), cmap='viridis', origin='lower', )
    # plt.imsave(f'temp/0.png', stacked)

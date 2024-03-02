import json
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import torchaudio
import soundfile as sf


class EpisodeDataset(Dataset):
    def __init__(self, data_folder="data"):
        """
        neg_ratio: ratio of silence audio clips to sample
        """
        super().__init__()
        # self.logs = pd.read_csv(log_file)
        self.data_folder = data_folder
        self.sr = 48000 ##TODO
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=int(self.sr * 0.025),
            hop_length=int(self.sr * 0.01),
            n_mels=64,
            center=False,
        )
        self.streams = [
            "cam_gripper",]
        # pass

    def get_episode(self, idx, ablation=""):
        """
        Return:
            folder for trial
            logs
            audio tracks
            number of frames in episode
        """
        modes = ablation.split("_")

        def load(file):
            fullpath = os.path.join(trial, file)
            if os.path.exists(fullpath):
                return sf.read(fullpath)[0]
            else:
                return None

        # format_time = self.logs.iloc[idx].Time.replace(":", "_")
        eps_ = 'run' + str(idx)
        # print("override" + '#' * 50)
        trial = os.path.join(self.data_folder, eps_)
        with open(os.path.join(trial, "ee_pose.json")) as ts:
            timestamps = json.load(ts)
        if "ag" in modes:
            audio_gripper1 = load(os.path.join(trial, "audio", "audio.wav"))

            # audio_gripper_left = audio_gripper1[:,0]
            # audio_gripper_right = audio_gripper1[:,1]
            # audio_gripper = [
            #     x for x in [audio_gripper_left, audio_gripper_right] if x is not None
            # ]
            # audio_gripper = torch.as_tensor(np.stack(audio_gripper, 0)).reshape(-1,2)
            # print(audio_gripper1.shape) #(434176, 2)
            audio_gripper = [
                x for x in audio_gripper1 if x is not None
            ]
            # print("ag", torch.tensor(audio_gripper).shape) #(434176, 2)
            audio_gripper = torch.as_tensor(np.stack(audio_gripper, 0))
            audio_gripper = (audio_gripper.T[0,:]).reshape(1,-1)
            print("ag1", audio_gripper.shape) #(434176, 2)
        else:
            audio_gripper = None
        # print(len(timestamps["action_history"])) # 4297
 
        return (
            trial,
            timestamps,
            audio_gripper,
            92, ##hardcoded
            # len(timestamps["action_history"]), TODO
        )

    def __getitem__(self, idx):
        raise NotImplementedError

    @staticmethod
    def load_image(trial, stream, timestep, leading_zeros=5):
        """
        Args:
            trial: the folder of the current episode
            stream: ["cam_gripper_color", "cam_fixed_color", "left_gelsight_frame"]
                for "left_gelsight_flow", please add another method to this class using torch.load("xxx.pt")
            timestep: the timestep of frame you want to extract
        """
        if timestep == -1:
            timestep = 0
        # print(timestep)
        img_path = os.path.join(trial, stream, str(timestep+1).zfill(leading_zeros) + ".png")
        image = (
            torch.as_tensor(np.array(Image.open(img_path))).float().permute(2, 0, 1)
            / 255
        )
        return image

    # @staticmethod
    # def load_flow(trial, stream, timestep):
    #     """
    #     Args:
    #         trial: the folder of the current episode
    #         stream: ["cam_gripper_color", "cam_fixed_color", "left_gelsight_frame"]
    #             for "left_gelsight_flow", please add another method to this class using torch.load("xxx.pt")
    #         timestep: the timestep of frame you want to extract
    #     """
    #     img_path = os.path.join(trial, stream, str(timestep) + ".pt")
    #     image = torch.as_tensor(torch.load(img_path))
    #     return image

    @staticmethod
    def clip_resample(audio, audio_start, audio_end):
        left_pad, right_pad = torch.Tensor([]), torch.Tensor([])
        print("au_size", audio.size())
        if audio_start < 0:
            left_pad = torch.zeros((audio.shape[0], -audio_start))
            audio_start = 0
        if audio_end >= audio.size(-1):
            right_pad = torch.zeros((audio.shape[0], audio_end - audio.size(-1)))
            audio_end = audio.size(-1)
        audio_clip = torch.cat(
            [left_pad, audio[:, audio_start:audio_end], right_pad], dim=1
        )
        print(f"start {audio_start}, end {audio_end} left {left_pad.size()}, right {right_pad.size()}. audio_clip {audio_clip.size()}")
        audio_clip = torchaudio.functional.resample(audio_clip, 48000, 16000)
        print(audio_clip.shape)
        
        return audio_clip

    def __len__(self):
        return len(self.logs)

    @staticmethod
    def resize_image(image, size):
        assert len(image.size()) == 3  # [3, H, W]
        return torch.nn.functional.interpolate(
            image.unsqueeze(0), size=size, mode="bilinear"
        ).squeeze(0)
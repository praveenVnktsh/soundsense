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

    def get_episode(self, idx, args, train, ablation=""):
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
            # print("fullpath", fullpath)
            if os.path.exists(fullpath):
                return sf.read(fullpath)[0]
            else:
                return None

        # # format_time = self.logs.iloc[idx].Time.replace(":", "_")
        # eps_ = 'run' + str(idx)
        # # print("override" + '#' * 50)
        # trial = os.path.join(self.data_folder, eps_)
        # with open(os.path.join(trial, "2024-02-27_19-40-16.json")) as ts:
        #     timestamps = json.load(ts)
        # timestamps = timestamps[::3]

        if train:
            train_csv = pd.read_csv(args.train_csv, header=None)[0]
            eps_ = str(train_csv.iloc[idx])
            with open(os.path.join("/home/punygod_admin/SoundSense/soundsense/data/mulsa/data", eps_ + "/actions.json")) as ts:
                timestamps = json.load(ts)
        else:
            val_csv = pd.read_csv(args.val_csv, header=None)[0]
            eps_ = str(val_csv.iloc[idx])
            with open(os.path.join("/home/punygod_admin/SoundSense/soundsense/data/mulsa/data", eps_ + "/actions.json")) as ts:
                timestamps = json.load(ts)

        trial = os.path.join(self.data_folder, eps_)
        num_img_frames = len(os.listdir(os.path.join(trial, "video")))
        # timestamps = timestamps[::3]

        if "ag" in modes:
            audio_gripper1 = load(os.path.join("processed_audio.wav"))

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
            # print("ag1", audio_gripper.shape) #(434176, 2)
            # audio_gripper = (audio_gripper.T[0,:]).reshape(1,-1)
            audio_gripper = (audio_gripper).reshape(1,-1)
            # print("ag1", audio_gripper.shape) #(434176, 2)
        else:
            audio_gripper = None
        # print(len(timestamps["action_history"])) # 4297
 
        return (
            trial,
            timestamps,
            audio_gripper,
            min(len(timestamps), num_img_frames) ##hardcoded
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
        img_path = os.path.join(trial, stream, str(timestep+1).zfill(leading_zeros ) + ".png")
        image = (
            torch.as_tensor(np.array(Image.open(img_path))).float().permute(2, 0, 1)
            / 255
        )
        # print(image.shape)
        # image = image[:,:,image.shape[2]//2:]
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
        audio_clip = torchaudio.functional.resample(audio_clip, 48000, 16000)
        # print("Inside clip_resample output shape", audio_clip.shape)
        
        return audio_clip

    def __len__(self):
        return len(self.logs)

    @staticmethod
    def resize_image(image, size):
        assert len(image.size()) == 3  # [3, H, W]
        return torch.nn.functional.interpolate(
            image.unsqueeze(0), size=size, mode="bilinear"
        ).squeeze(0)

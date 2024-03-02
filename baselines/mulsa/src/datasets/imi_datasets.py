import os
import torch
import torchvision.transforms as T

from src.datasets.base import EpisodeDataset
import numpy as np
from PIL import Image
import random

class ImitationEpisode(EpisodeDataset):
    def __init__(self, args, dataset_idx, data_folder, train=True):
        # print("d_idx", dataset_idx)
        super().__init__( data_folder)
        self.train = train
        self.num_stack = args.num_stack
        self.frameskip = args.frameskip
        self.max_len = (self.num_stack - 1) * self.frameskip
        self.fps = 10
        self.sr = 48000 ##44100 TODO
        self.resolution = (
            self.sr // self.fps
        )  # number of audio samples in one image idx
        # self.audio_len = int(self.resolution * (max(self.max_len + 1, 10)))
        self.audio_len = self.num_stack * self.frameskip * self.resolution

        # augmentation parameters
        self.EPS = 1e-8
        self.resized_height_v = args.resized_height_v
        self.resized_width_v = args.resized_width_v
        self.resized_height_t = args.resized_height_t
        self.resized_width_t = args.resized_width_t
        self._crop_height_v = int(self.resized_height_v * (1.0 - args.crop_percent))
        self._crop_width_v = int(self.resized_width_v * (1.0 - args.crop_percent))
        self._crop_height_t = int(self.resized_height_t * (1.0 - args.crop_percent))
        self._crop_width_t = int(self.resized_width_t * (1.0 - args.crop_percent))
        (
            self.trial,
            self.timestamps,
            self.audio_gripper,
            self.num_frames,
        ) = self.get_episode(dataset_idx, ablation=args.ablation)

        # # saving the offset for gelsight in order to normalize data
        # self.gelsight_offset = (
        #     torch.as_tensor(
        #         np.array(Image.open(os.path.join(self.data_folder, "gs_offset.png")))
        #     )
        #     .float()
        #     .permute(2, 0, 1)
        #     / 255
        # )
        self.action_dim = args.action_dim
        self.task = args.task
        self.modalities = args.ablation.split("_")
        self.nocrop = args.nocrop

        if self.train:
            self.transform_cam = [
                T.Resize((self.resized_height_v, self.resized_width_v)),
                T.ColorJitter(brightness=0.2, contrast=0.02, saturation=0.02),
            ]
            self.transform_cam = T.Compose(self.transform_cam)

        else:
            self.transform_cam = T.Compose(
                [
                    T.Resize((self.resized_height_v, self.resized_width_v)),
                    T.CenterCrop((self._crop_height_v, self._crop_width_v)),
                ]
            )

    def __len__(self):
        return self.num_frames

    def get_demo(self, idx):
        # TODO: Discretize action space if not already
        # print(len(self.timestamps["action_history"]))
        keyboard = self.timestamps["action_history"][idx]
        # print(keyboard)
        if self.task == "pouring":
            x_space = {-0.0005: 0, 0: 1, 0.0005: 2}
            dy_space = {-0.0012: 0, 0: 1, 0.004: 2}
            keyboard = x_space[keyboard[0]] * 3 + dy_space[keyboard[4]]
        else:
            x_space = {-0.0005: 0, 0: 1, 0.0005: 2}
            y_space = {-0.0005: 0, 0: 1, 0.0005: 2}
            z_space = {-0.0005: 0, 0: 1, 0.0005: 2}
            keyboard = (
                x_space[keyboard[0]] * 9
                + y_space[keyboard[1]] * 3
                + z_space[keyboard[2]]
            )
        return keyboard

    def __getitem__(self, idx):
        print("idx", idx, self.max_len)
        start = idx - self.max_len
        print("start", start)
        # compute which frames to use TODO
        frame_idx = np.arange(start, idx + 1, self.frameskip)
        frame_idx[frame_idx < 0] = -1
        # images
        # to speed up data loading, do not load img if not using
        cam_gripper_framestack = 0
        # print(frame_idx)
        # process different streams of data
        if "vg" in self.modalities:
            cam_gripper_framestack = torch.stack(
                [
                    self.transform_cam(
                        self.load_image(self.trial, "frames", timestep)
                    )
                    for timestep in frame_idx
                ],
                dim=0,
            )

        # random cropping
        if self.train:
            # print("idx1", idx)
            img = self.transform_cam(
                self.load_image(self.trial, "frames", idx)
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

        # load audio
        # audio_end = idx * self.resolution
        # audio_start = audio_end - self.audio_len  # why self.sr // 2, and start + sr
        print("idx..", idx)
        audio_start = int((idx/10-3)*self.sr)
        audio_end = int((idx/10)*self.sr)
        print(self.audio_len, audio_end, audio_start, self.resolution)
        if self.audio_gripper is not None:
            audio_clip_g = self.clip_resample(
                self.audio_gripper, audio_start, audio_end
            ).float()
            print("ag", audio_clip_g.shape)
        else:
            audio_clip_g = 0
 
        # load labels ## TODO
        keyboard = random.randint(0, 2)  # self.get_demo(idx)
        xyzrpy =  torch.Tensor(self.timestamps["action_history"][idx][:6])
        # torch.Tensor(self.timestamps["pose_history"][idx][:6])

        return (
            (cam_gripper_framestack,
            audio_clip_g,),
            keyboard,
            xyzrpy,
            start,
        )


# class TransformerEpisode(ImitationEpisode):
#     @staticmethod
#     def load_image(trial, stream, timestep):
#         """
#         Do not duplicate first frame for padding, instead return all zeros
#         """
#         return_null = timestep == -1
#         if timestep == -1:
#             timestep = 0
#         img_path = os.path.join(trial, stream, str(timestep) + ".png")
#         image = (
#             torch.as_tensor(np.array(Image.open(img_path))).float().permute(2, 0, 1)
#             / 255
#         )
#         if return_null:
#             image = torch.zeros_like(image)
#         return image

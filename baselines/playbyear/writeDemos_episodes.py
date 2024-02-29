import platform

import torchaudio
print(platform.node())
import copy
import math
import os
import pickle as pkl
import sys
import cv2
import soundfile as sf
from tqdm import tqdm

import time

from shutil import copyfile

import numpy as np

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import core.utils as utils
# from core.logger import Logger
from core.replay_buffer_3 import ReplayBufferDoubleRewardEpisodes as ReplayBuffer
from core.replay_buffer_audio_episode import ReplayBufferAudioEpisodes as ReplayAudioBuffer
# from core.video import VideoRecorder
import gym
import csv

from robosuite import make
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.controllers import load_controller_config

torch.backends.cudnn.benchmark = True

# from custom_environments.indicatorboxBlock import IndicatorBoxBlock
# from custom_environments.blocked_pick_place import BlockedPickPlace

IMG_HEIGHT = 84
IMG_WIDTH = 84

def debug(inp):
    if type(inp) is not str:
        inp = str(inp)
    print("\033[33mDEBUG: "+inp+"\033[0m")


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        print("cuda status: ", torch.cuda.is_available())
        debug(f"audio: {cfg.audio}")
        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        # self.env = make_env(cfg)

        
        if self.cfg.audio:
            self.replay_buffer = ReplayAudioBuffer((1,130),
                                                    (9, IMG_WIDTH, IMG_HEIGHT),
                                                    (57,160), (4,),
                                                    self.cfg.episodes,
                                                    self.cfg.episodeLength,
                                                    self.cfg.image_pad, self.device)
        else:
            self.replay_buffer = ReplayBuffer((1, 130),                  # TODO: Hardcoded so change
                                            (9, IMG_WIDTH, IMG_HEIGHT),
                                            (4,),
                                            self.cfg.episodes,
                                            self.cfg.episodeLength,
                                            self.cfg.image_pad, self.device)
            

        # self.video_recorder = VideoRecorder(
        #     self.work_dir if cfg.save_video else None)
        self.step = 3

    def get_image_frames(self, dir_path, idx):
        stacked_imgs = np.zeros((9, IMG_WIDTH, IMG_HEIGHT))                     # TODO: Hardcoded

        for i in range(3):
            img_num = f'{idx-i:05d}.png'
            img_path = os.path.join(dir_path, img_num)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img[:, 1280:, :], (IMG_WIDTH, IMG_HEIGHT))
            stacked_imgs[3*i:3*i+3, :, :] = img_resized.reshape(3, IMG_WIDTH, IMG_HEIGHT)

        return stacked_imgs
    
    def clip_resample(self, audio, audio_start, audio_end):
        audio = torch.from_numpy(audio.T)
        # debug(f"audio size() {audio.size()}")
        left_pad, right_pad = torch.Tensor([]), torch.Tensor([])
        if audio_start < 0:
            left_pad = torch.zeros((audio.shape[0], -audio_start))
            audio_start = 0
        if audio_end >= audio.size(-1):
            right_pad = torch.zeros((audio.shape[0], audio_end - audio.size(-1)))
            audio_end = audio.size(-1)
        audio_clip = torch.cat(
            [left_pad, audio[:, audio_start:audio_end], right_pad], dim=1
        )
        # debug(f"start {audio_start}, end {audio_end} left {left_pad.size()}, right {right_pad.size()}. audio_clip {audio_clip.size()}")
        audio_clip = torchaudio.functional.resample(audio_clip, 48000, 16000)
        return audio_clip

    def get_audio_frames(self, waveforms, idx, sr=48000, mel_sr=16000):
        # debug(f"audio size() {waveforms.size}")
        waveform = self.clip_resample(waveforms, int((idx/10-2)*sr), int((idx/10)*sr))
        # debug(f"waveform shape {waveform.shape}")
        mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=mel_sr, n_fft=int(mel_sr * 0.025), hop_length=1+int(mel_sr * 0.025)//2, n_mels=57
            )
        # Original hop_length = int(mel_sr * 0.01)
        EPS = 1e-8
        spec = mel(waveform.float())
        log_spec = torch.log(spec + EPS)
        # debug(f"log_spec.size() {log_spec.size()}")
        assert log_spec[0].size() == (57, 160), f"audio spec size mismatch. expected (57, 160), got {log_spec[0].size()}"
        return log_spec[0]     # TODO: return only 1 channel for now
        # if self.norm_audio:
        #     log_spec /= log_spec.sum(dim=-2, keepdim=True)  # [1, 64, 100]
    
    # frame number = time * 10
    # time = frame number / 10
    # self.step is frame number
    # time = self.step / 10
    # ((self.step/10)-2)*sr to (self.step/10)*sr


    def real_data(self, cfg, img_path, audio_path):
        buffer_list = list()

        episodeLength = len(os.listdir(img_path))
        # episodeLength = 4
        waveforms = sf.read(audio_path)[0]
        for ep in range(self.cfg.episodes):
            self.step = 3
            # while self.step < cfg.episodeLength:
            pbar = tqdm(total=episodeLength-self.step)
            while self.step < episodeLength:
                # obs = np.random.randint(low=0, high=255, size=(9, 84, 84)).astype('uint8')
                # action = np.random.rand(4)
                
                obs = self.get_image_frames(img_path, self.step).astype('uint8')
                # debug(f"{self.step}, {audio_obs.size()}")
                # debug(f"Loaded images at step {self.step}")
                action = np.random.rand(4)

                # Random because we don't use it
                lowdim = np.random.rand(130)
                reward = 0.
                next_lowdim = np.random.rand(130)
                next_obs = np.random.randint(low=0, high=255, size=(9, IMG_WIDTH, IMG_HEIGHT)).astype('uint8')
                done = 0.
                done_no_max = 0.
                # debug(f"lowdim.shape {lowdim.shape}\n obs.shape {obs.shape}\n action.shape {action.shape}\n reward.shape {reward}\n next_lowdim.shape {next_lowdim.shape}\n next_obs.shape {next_obs.shape}\n done {done}\n done_no_max {done_no_max}")
                if cfg.audio:
                    audio_obs = self.get_audio_frames(waveforms, self.step)
                    next_audio = np.random.rand(57, 160)
                    buffer_list.append((lowdim, obs, audio_obs, action, reward,
                                    (1.0 if self.step > cfg.sparseProp * cfg.episodeLength else 0.0), next_lowdim, next_obs, next_audio, done, done_no_max))
                else:
                    buffer_list.append((lowdim, obs, action, reward,
                                    (1.0 if self.step > cfg.sparseProp * cfg.episodeLength else 0.0), next_lowdim, next_obs, done, done_no_max))


                self.step += 1
                pbar.update(1)
            pbar.close()
            self.replay_buffer.add(buffer_list)
            print("****** ADDED ****** and we are at ", self.replay_buffer.idx)


    def run(self, cfg):
        img_path = '/home/punygod_admin/SoundSense/soundsense/data/run3/frames'
        audio_path = '/home/punygod_admin/SoundSense/soundsense/data/run3/audio/audio.wav'
        self.real_data(cfg=cfg, img_path=img_path, audio_path=audio_path)
        write_path = "/home/punygod_admin/SoundSense/soundsense/data/run3/pbe"
        if cfg.audio:
            write_path+="_audio"
        write_path+=".pkl"
        pkl.dump(self.replay_buffer, open(write_path, "wb" ), protocol=4 )
        print("Demos saved")


@hydra.main(config_path='writeDemos_episodes.yaml', strict=True)
def main(cfg):
    from writeDemos_episodes import Workspace as W
    workspace = W(cfg)
    workspace.run(cfg)

if __name__ == '__main__':
    main()


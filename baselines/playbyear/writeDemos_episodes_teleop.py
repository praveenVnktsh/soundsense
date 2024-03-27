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
import json
import glob
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

        if ~self.cfg.append_mode:
            if self.cfg.audio:
                self.replay_buffer = ReplayAudioBuffer(
                                                        (9, IMG_WIDTH, IMG_HEIGHT),
                                                        (57,160), (11,),
                                                        self.cfg.episodes,
                                                        self.cfg.episodeLength,
                                                        self.cfg.image_pad, self.device)
            else:
                self.replay_buffer = ReplayBuffer(                  # TODO: Hardcoded so change
                                                (9, IMG_WIDTH, IMG_HEIGHT),
                                                (11,),
                                                self.cfg.episodes,
                                                self.cfg.episodeLength,
                                                self.cfg.image_pad, self.device)
        else:
            self.replay_buffer = pkl.load(open(self.cfg.append_file, "rb"))

        # self.video_recorder = VideoRecorder(
        #     self.work_dir if cfg.save_video else None)
        self.step = 3

    def get_image_frames(self, paths):
        stacked_imgs = np.zeros((9, IMG_WIDTH, IMG_HEIGHT))                     # TODO: Hardcoded

        for i, path in enumerate(paths):
            
            
            img = cv2.imread(path)
            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            stacked_imgs[3*i:3*i+3, :, :] = img_resized.reshape(3, IMG_WIDTH, IMG_HEIGHT)

        return stacked_imgs
    
    def clip_resample(self, audio, audio_start, audio_end):
        audio = torch.from_numpy(audio.T)
        # # debug(f"audio size() {audio.size()}")
        left_pad, right_pad = torch.Tensor([]), torch.Tensor([])
        if audio_start < 0:
            
            left_pad = torch.zeros((-audio_start, ))
            audio_start = 0
        
        if audio_end >= audio.size(-1):
            right_pad = torch.zeros((audio_end - audio.size(-1), ))
            audio_end = audio.size(-1)

        
        audio_clip = torch.cat(
            [left_pad, audio[audio_start:audio_end], right_pad], dim=0
        )

        # debug(f"start {audio_start}, end {audio_end} left {left_pad.size()}, right {right_pad.size()}. audio_clip {audio_clip.size()}")
        audio_clip = audio_clip.unsqueeze(0)
        audio_clip = torchaudio.functional.resample(audio_clip, 48000, 16000)
        return audio_clip

    def get_audio_frames(self, waveforms, time_from_start, prev_seconds = 2, sr=48000, mel_sr=16000):
        debug(f"audio size() {waveforms.size}")
        waveform = self.clip_resample(waveforms, int((time_from_start - prev_seconds)*sr), int((time_from_start)*sr))
        
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=mel_sr, n_fft=int(mel_sr * 0.025), hop_length=1+int(mel_sr * 0.025)//2, n_mels=57
        )
        # Original hop_length = int(mel_sr * 0.01)
        EPS = 1e-8
        spec = mel(waveform.float())
        log_spec = torch.log(spec + EPS)
        
        debug(f"log_spec.size() {log_spec.size()}")
        assert log_spec[0].size() == (57, 160), f"audio spec size mismatch. expected (57, 160), got {log_spec[0].size()}"
        return log_spec[0]     # TODO: return only left channel for now

    
    # frame number = time * 10
    # time = frame number / 10
    # self.step is frame number
    # time = self.step / 10
    # ((self.step/10)-2)*sr to (self.step/10)*sr


    def real_data(self, cfg, root_dir):
        runs = sorted(glob.glob(root_dir+"*/"))
        for run in runs:
            print("At run: ", run)
            buffer_list = list()
            images_path = run + "video/"
            audio_path  = run + "processed_audio.wav"
            # 1711051394806730

            action_path = run + "actions.json"
            episodeLength = len(glob.glob(images_path+"*.png"))
            
            waveforms = sf.read(audio_path)[0]

            action_np = np.array(json.load(open(action_path)))
            if len(action_np) == 0:
                continue
            # For now since lengths are different
            actions_idx = np.arange(min(action_np.shape[0], episodeLength*30//10), step=3)
            action_np = action_np[actions_idx, :]    
            episodeLength = min(episodeLength, action_np.shape[0])

            self.step = 3
            pbar = tqdm(total=episodeLength-self.step)
            image_paths = sorted(glob.glob(images_path+"*.png"))
            starttime = int(os.path.basename(image_paths[0]).split(".")[0])/1e6

            while self.step < episodeLength:
                
                obs = self.get_image_frames(image_paths[self.step - 3 : self.step]).astype('uint8')
                action = action_np[self.step]
                reward = 0.
                next_obs = np.random.randint(low=0, high=255, size=(9, IMG_WIDTH, IMG_HEIGHT)).astype('uint8')
                done = 0.
                done_no_max = 0.
                if cfg.audio:
                    # current timestamp
                    cur_time = int(os.path.basename(image_paths[self.step]).split(".")[0])/1e6
                    delta_from_start = cur_time - starttime # in seconds
                    print(delta_from_start)
                    audio_obs = self.get_audio_frames(waveforms, delta_from_start, prev_seconds = 2)
                    next_audio = np.random.rand(57, 160)
                    buffer_list.append((obs, audio_obs, action, reward,
                                    (1.0 if self.step > cfg.sparseProp * cfg.episodeLength else 0.0), next_obs, next_audio, done, done_no_max))
                else:
                    # print("action saved: ", action)   
                    buffer_list.append((obs, action, reward,
                                    (1.0 if self.step > cfg.sparseProp * cfg.episodeLength else 0.0), next_obs, done, done_no_max))
                self.step += 1
                pbar.update(1)
            pbar.close()
            self.replay_buffer.add(buffer_list)
            self.replay_buffer.end_add()
            print("****** ADDED ****** and we are at ", self.replay_buffer.idx)


    def run(self, cfg):
        root_dir = '/home/punygod_admin/SoundSense/soundsense/data/seventysix/data/'
        self.real_data(cfg, root_dir)
        write_path = "/home/punygod_admin/SoundSense/soundsense/data/playbyear_pkls/pbe"
        if cfg.audio:
            write_path+="_audio"
        write_path+=f"_{cfg.episodes}.pkl"
        pkl.dump(self.replay_buffer, open(write_path, "wb" ), protocol=4 )
        print("Demos saved at ", write_path)


@hydra.main(config_path='writeDemos_episodes.yaml', strict=True)
def main(cfg):
    from writeDemos_episodes_teleop import Workspace as W
    workspace = W(cfg)
    print("Starting")
    workspace.run(cfg)

if __name__ == '__main__':
    main()


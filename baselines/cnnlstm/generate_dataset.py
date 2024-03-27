import torchaudio
import os
import pickle as pkl
import sys
import cv2
import soundfile as sf
from tqdm import tqdm
import json
import glob
from shutil import copyfile

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# torch.backends.cudnn.benchmark = True

# from custom_environments.indicatorboxBlock import IndicatorBoxBlock
# from custom_environments.blocked_pick_place import BlockedPickPlace

IMG_HEIGHT = 224
IMG_WIDTH = 224

def debug(inp):
    if type(inp) is not str:
        inp = str(inp)
    print("\033[33mDEBUG: "+inp+"\033[0m")


class Workspace():
    def __init__(self, audio):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        print("cuda status: ", torch.cuda.is_available())
        self.buffer = []
        self.n_stack = 1
        self.step = self.n_stack
        self.use_audio = audio

    def get_image_frames(self, paths):
        stacked_imgs = np.zeros((self.n_stack, 3, IMG_WIDTH, IMG_HEIGHT))
        for i, path in enumerate(paths):
            img = cv2.imread(path)
            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            stacked_imgs[i * self.n_stack:self.n_stack*(i+1), :, :] = img_resized.reshape(3, IMG_WIDTH, IMG_HEIGHT)
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
        # debug(f"audio size() {waveforms.size}")
        waveform = self.clip_resample(waveforms, int((time_from_start - prev_seconds)*sr), int((time_from_start)*sr))
        
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=mel_sr, n_fft=int(mel_sr * 0.025), hop_length=1+int(mel_sr * 0.025)//2, n_mels=57
        )

        # Original hop_length = int(mel_sr * 0.01)
        EPS = 1e-8
        spec = mel(waveform.float())
        log_spec = torch.log(spec + EPS)
        
        # debug(f"log_spec.size() {log_spec.size()}")
        assert log_spec[0].size() == (57, 160), f"audio spec size mismatch. expected (57, 160), got {log_spec[0].size()}"
        return log_spec[0]     # TODO: return only left channel for now

    
    # frame number = time * 10
    # time = frame number / 10
    # self.step is frame number
    # time = self.step / 10
    # ((self.step/10)-2)*sr to (self.step/10)*sr


    def real_data(self, root_dir):
        runs = sorted(glob.glob(root_dir+"*/"))
        n_bufs = 0
        for i, run in enumerate(runs):
            if i < 40:
                continue
            print("Processing ", run)
            images_path = run + "video/"
            audio_path  = run + "processed_audio.wav"
            action_path = run + "actions.json"
            episodeLength = len(glob.glob(images_path+"*.png"))
            waveforms = sf.read(audio_path)[0]
            action_np = np.array(json.load(open(action_path)))
            # For now since lengths are different
            # actions_idx = np.arange(min(action_np.shape[0], episodeLength*30//10), step=3)
            # action_np = action_np[actions_idx, :]
            episodeLength = min(episodeLength, action_np.shape[0])
            pbar = tqdm(total=episodeLength-self.step)
            image_paths = sorted(glob.glob(images_path+"*.png"))
            starttime = int(os.path.basename(image_paths[0]).split(".")[0])/1e6
            self.step = self.n_stack
            while self.step < episodeLength:
                obs = self.get_image_frames(image_paths[self.step - self.n_stack : self.step]).astype('uint8')
                action = action_np[self.step]
                # current timestamp
                cur_time = int(os.path.basename(image_paths[self.step]).split(".")[0])/1e6
                delta_from_start = cur_time - starttime # in seconds
                try:
                    audio_obs = self.get_audio_frames(waveforms, delta_from_start, prev_seconds = 2)
                except:
                    continue
                self.buffer.append((
                        audio_obs,
                        obs,
                        action,
                    ))
                self.step += 1
                pbar.update(1)
            pbar.close()
            
            print("We have ", len(self.buffer), " samples")
            if self.use_audio:
                stringg = 'audio'
            else:
                stringg = 'video'
            write_path = f"/home/punygod_admin/SoundSense/soundsense/data/cnn_baseline_data/{stringg}/data_{n_bufs}.pkl"
            print("Writing Buffer of length = ", len(self.buffer))
            pkl.dump(self.buffer, open(write_path, "wb" ), protocol=4 )
            n_bufs += 1
            self.buffer = []



    def run(self):
        os.makedirs('/home/punygod_admin/SoundSense/soundsense/data/cnn_baseline_data/', exist_ok=True)
        root_dir = '/home/punygod_admin/SoundSense/soundsense/data/seventysix/data/'
        self.real_data(root_dir)

def main():
    
    workspace = Workspace(True)
    print("Starting")
    workspace.run()

if __name__ == '__main__':
    main()


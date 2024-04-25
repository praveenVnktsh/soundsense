import numpy as np
import torch
import torchaudio

class AudioProcessor:

    def __init__(self, config):
        self.resample_rate_audio = config["resample_rate_audio"] 
        self.sample_rate_audio  = config["sample_rate_audio"]
        self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.resample_rate_audio, 
                n_fft=512, 
                hop_length=int(self.resample_rate_audio * 0.01), 
                n_mels=64
            )
        self.audio_encoder = config["audio_encoder"] if "audio_encoder" in config else "spec"
        self.norm_audio = config["norm_audio"] if "norm_audio" in config else False

    def process(self, audio, audio_start, audio_end, clip_and_resample = False):
        if clip_and_resample:
            audio = self.clip_and_resample(audio, audio_start, audio_end, )
        
        mel = self.mel(audio.to(torch.float32))
        eps = 1e-8
        mel = np.log(mel + eps)
        if self.norm_audio:
            mel /= mel.sum(dim=-2, keepdim=True)

        # import soundfile as sf
        # import matplotlib.pyplot as plt
        # sf.write(f'temp/audio.wav', audio[0].numpy(), self.resample_rate_audio)
        # plt.imsave('temp/mel.png', mel[0].numpy(), cmap='viridis', origin='lower', )
        # print("RESAMPLEd", audio.min(), audio.max(), audio.mean(), audio.std())
        # print("AUDIO GRIPPER", self.audio_gripper.min(), self.audio_gripper.max(), self.audio_gripper.mean(), self.audio_gripper.std())
        # exit()

        
        return mel
    
    def clip_and_resample(self, audio, audio_start, audio_end):
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
        if self.sample_rate_audio != self.resample_rate_audio:
            audio_clip = torchaudio.functional.resample(
                audio_clip, 
                self.sample_rate_audio,
                self.resample_rate_audio, 
                audio.shape[-1]
            )
        return audio_clip.float()
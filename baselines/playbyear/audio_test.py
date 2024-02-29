import soundfile as sf
import torch
import torchaudio
import matplotlib.pyplot as plt

def debug(inp):
    if type(inp) is not str:
        inp = str(inp)
    print("\033[33mDEBUG: "+inp+"\033[0m")

def clip_resample(audio, audio_start, audio_end):
    audio = torch.from_numpy(audio.T)
    debug(f"audio size() {audio.size()}")
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
    audio_clip = torchaudio.functional.resample(audio_clip, 48000, 16000)
    return audio_clip

def get_audio_frames(waveforms, idx, sr=48000, mel_sr=16000):
    debug(f"audio size() {waveforms.size}")
    waveform = clip_resample(waveforms, (idx-2)*sr, idx*sr)
    debug(f"waveform shape {waveform.shape}")
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=mel_sr, n_fft=int(mel_sr * 0.025), hop_length=1+int(mel_sr * 0.025)//2, n_mels=57
    )
    # n_fft = int(mel_sr * 0.025)
    EPS = 1e-8
    spec = mel(waveform.float())
    log_spec = torch.log(spec + EPS)
    debug(f"log_spec.size() {log_spec.size()}")
    # assert log_spec.size(-2) == 64
    return log_spec
    # if self.norm_audio:
    #     log_spec /= log_spec.sum(dim=-2, keepdim=True)  # [1, 64, 100]

path = "/home/punygod_admin/SoundSense/soundsense/data/run3/audio/audio.wav"
audio_spec = get_audio_frames(sf.read(path)[0], 3)
debug(f"audio_spec.shape {audio_spec.shape}\n audio min {audio_spec[0].min()} audio max {audio_spec[0].max()}\n {audio_spec[0,1,:]}")


plt.imshow(audio_spec[0])
plt.colorbar()
plt.savefig("audio_spec.png")
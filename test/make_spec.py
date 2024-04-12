import wave
import numpy as np
import torch, torchaudio
import matplotlib.pyplot as plt
import soundfile as sf
hz = 16000
with wave.open('output.wav', 'r') as wf:
    data = wf.readframes(hz * 10)
    wav_data = np.frombuffer(data, dtype=np.int16)#.astype(np.float32)
    print(wav_data.min(), wav_data.max())

audio_data, hz = sf.read('output.wav')

print(audio_data.shape, hz)
print(audio_data.max(), audio_data.min())
melo = torchaudio.transforms.MelSpectrogram(
    sample_rate=hz, 
    n_fft=512, 
    hop_length=int(hz* 0.01), 
    n_mels=64
)
eps = 1
mm = melo(torch.tensor(audio_data).float())
mel = np.log(mm + eps)
plt.imsave("melspec.png", mm.numpy(), cmap = 'viridis',origin='lower',)
plt.imsave('logmelspec.png', mel.numpy(), cmap='viridis', origin='lower', )

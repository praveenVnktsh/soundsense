import rospy 
import soundfile as sf
from audio_common_msgs.msg import AudioData
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import torch
import time
buffer = []
dirbuf = []
hz = 16000
starttime = None
def callback(data):
    global buffer, hz, starttime, dirbuf

    if starttime is None:
        starttime = time.time()
    audio = np.frombuffer(data.data, dtype=np.int16).tolist()
    buffer += (audio.copy())
    # print(len(buffer))

rospy.init_node("temp")
audio_sub = rospy.Subscriber('/audio/audio', AudioData, callback)
rospy.sleep(5)
audio_sub.unregister()

print("Audio recorded for ", time.time() - starttime, " seconds")
origbuf = np.array(buffer.copy(), dtype = np.int16)
buffer = np.array(buffer, dtype = np.float64).flatten()
buffer /= 32768
print(buffer.max(), buffer.min())
print(len(buffer))
print("Time of buffer: ", len(buffer) / hz)


melo = torchaudio.transforms.MelSpectrogram(
    sample_rate=hz, 
    n_fft=512, 
    hop_length=int(hz* 0.01), 
    n_mels=64
)
eps = 1
mm = melo(torch.tensor(buffer).float())
mel = np.log( mm + eps)
print(mel.min(), mel.max())
plt.imsave("melspec.png", mm.numpy(), cmap = 'viridis',origin='lower',)
plt.imsave('logmelspec.png', mel.numpy(), cmap='viridis', origin='lower', )


import wave

with wave.open('output.wav', 'wb') as wf:
    wf.setnchannels(1)  
    wf.setsampwidth(2)  
    wf.setframerate(hz)
    wf.writeframes(origbuf.tobytes())
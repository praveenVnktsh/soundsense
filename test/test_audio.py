import rospy 
import soundfile as sf
from audio_common_msgs.msg import AudioData
import numpy as np
import torchaudio
buffer = []
def callback(data):
    global buffer
    audio = np.frombuffer(data.data, dtype=np.uint8).tolist()
    buffer.append(audio)
    if len(buffer) > 16000 * 3:
        buffer = buffer[-16000 * 3:]
    
rospy.init_node("temp")
audio_sub = rospy.Subscriber('/audio/audio', AudioData, callback)
rospy.sleep(5)
audio_sub.unregister()

buffer = np.array(buffer, dtype = np.uint8).flatten()
print(buffer.max(), buffer.min())
# buffer -= 128
# buffer /= 128
print(buffer.max(), buffer.min())
print(len(buffer))
# sf.write("test.wav", buffer, 16000)
mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, 
        n_fft=512, 
        hop_length=int(16000* 0.01), 
        n_mels=64
    )

# melspec = np.log(mel(torch.tensor()) + 1)
# if self.norm_audio:
#     mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
#     mel -= mel.mean()
import wave

# with wave.open('output.wav', 'wb') as wf:
#     wf.setnchannels(1)  # mono audio
#     wf.setsampwidth(2)  # 16-bit audio
#     wf.setframerate(16000)
#     wf.writeframes(buffer.tobytes())
import sounddevice as sd
import numpy as np

fs=48000
duration = 5  # seconds
devices = sd.query_devices()
print(devices)
mic = devices[5]
print(mic)
sd.default.device = 5


# myrecording = sd.rec(duration * fs, samplerate=fs, channels=2,dtype='float64',)
# sd.wait()
# sd.play(myrecording, fs)
# sd.wait()
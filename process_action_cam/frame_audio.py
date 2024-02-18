import os
import sys
import cv2
import subprocess

## for the same timestamp capture the audio and frame from a video and save it in two different folders using ffmpeg
## the audio and the frame should have the same name

audio_dir = "data/run1/audio"
frame_dir = "data/run1/frames"

if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)

video_file = "data/run1/video.mp4"

## extract timestamps

vid_cap = cv2.VideoCapture(video_file)
timestamp_arr = []
while True:
    res, frame = vid_cap.read()
    if not res:
        break
    timestamp = vid_cap.get(cv2.CAP_PROP_POS_MSEC)
    timestamp_arr.append(timestamp)

# use ffmepeg to extract the audio and the frames

for t in range(0, len(timestamp_arr), 9):
    command = f"ffmpeg -i {video_file} -ss {timestamp_arr[t]/1000} -t 0.3 -vn {audio_dir}/{timestamp_arr[t]}.mp3"
    subprocess.call(command, shell=True)

    command = f'ffmpeg -i {video_file} -ss {timestamp_arr[t]/1000} -t 0.3 -vf "fps=1" {frame_dir}/{timestamp_arr[t]}.jpg'
    subprocess.call(command, shell=True)

### how to take history
### timestamp for the audio and the frame can be different? Alignment issue?

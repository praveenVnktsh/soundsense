import os
import sys
import subprocess
import ffmpeg
import datetime
import numpy as np


## for the same timestamp capture the audio and frame from a video and save it in two different folders using ffmpeg
## the audio and the frame should have the same name
root = '../data/'
run_id = 'run1' + '/'
audio_dir = root + run_id + "audio/"
frame_dir = root + run_id + "frames/"

if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)

video_file = root + run_id + "video.mp4"

## extract timestamps
# probe = ffmpeg.probe(video_file)
# metadata = probe["format"]["tags"]
# iso_time = metadata["creation_time"]
# dt_object = datetime.datetime.strptime(iso_time, "%Y-%m-%dT%H:%M:%S.%fZ")
# unix_timestamp = int(dt_object.timestamp())
# video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
# duration =  float(video_info['duration'])
# frame_rate = int(video_info['avg_frame_rate'].split('/')[0])/int(video_info['avg_frame_rate'].split('/')[1])



# use ffmepeg to extract the audio and the frames
save_audio = True
save_video = True
freq_rate = 10
leading_zeros = 5
if save_video:
    command = f'ffmpeg -i {video_file} -r {freq_rate} {frame_dir}/%0{leading_zeros}d.png'
    subprocess.call(command, shell=True)

# get total number of images
tot_images = len(os.listdir(frame_dir))

with open(root + run_id + 'timestamps.txt', 'w') as f:
    for i in range(tot_images):
        f.write(str(i/freq_rate) + '\tframes/' + str(i).zfill(leading_zeros) + '.png' + '\n')


if save_audio:
    command = f"ffmpeg -i {video_file} -vn {audio_dir}/audio.wav"
    subprocess.call(command, shell=True)

### how to take history
### timestamp for the audio and the frame can be different? Alignment issue?

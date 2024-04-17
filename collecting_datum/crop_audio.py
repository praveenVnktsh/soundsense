from pydub import AudioSegment
import glob
import os
from natsort import natsorted

# root_fold = '/home/hello-robot/soundsense/soundsense/stretch/data/data_two_cups'
root_fold = '/home/punygod_admin/SoundSense/soundsense/data/mulsa/data'
for run_id in natsorted(os.listdir(root_fold)):
    print("Processing ", run_id)
    root = f'{root_fold}/{run_id}'

    imgs = sorted(glob.glob(f'{root}/video/*.png'))
    start_timestamp = os.path.basename(imgs[0]).split('.')[0]
    end_timestamp = os.path.basename(imgs[-1]).split('.')[0]
    start_timestamp = int(start_timestamp) / 1e6
    end_timestamp = int(end_timestamp) / 1e6
    # print("Images start at", start_timestamp, 'and end at',  end_timestamp)

    audios = sorted(glob.glob(f'{root}/*.wav'))
    if len(audios) == 0:
        # print("No audio file found. Skipping", run_id)
        continue
    if len(audios) > 1:
        # print("Multiple audio files found. Filtering")
    
        audios = [audio for audio in audios if os.path.basename(audio).split('.')[0].isdigit()]


    audio_chosen = audios[-1]

    
    
    newAudio = AudioSegment.from_wav(audio_chosen)

    audio_start_timestamp = str(os.path.basename(audios[0]).split('.')[0])
    audio_start_timestamp = int(audio_start_timestamp) / 1e6
    # print("Audio starts at", audio_start_timestamp)

    crop_start = int((start_timestamp - audio_start_timestamp) * 1000)
    crop_end = int((end_timestamp - audio_start_timestamp) * 1000)
    # print("Cropping between ", crop_start, crop_end, 'ms')
    # print("Duration of audio =", (crop_end - crop_start)/1000)
    dur = (crop_end - crop_start)/1000
    if dur > 220:
        print("Skipping since its a long file. id =", run_id, 'duration =', dur)
        continue
    # print("Duration of audio is ", (crop_end - crop_start)/1000., "Duration of video is", end_timestamp - start_timestamp)
    # print()
    newAudio = newAudio[crop_start:crop_end]
    newAudio.export(str(os.path.dirname(audios[0])) + '/processed_audio.wav', format="wav") #Exports to a wav file in the current path.

    # exit()
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from IPython.display import Video

def compute_mel_spectrogram(audio_path, sr=48000, n_fft=2048, hop_length=512, n_mels=128):
    """
    Compute the mel spectrogram of an audio file.
    
    Parameters:
        audio_path (str): Path to the audio file.
        sr (int): Sampling rate of the audio file (default: 22050 Hz).
        n_fft (int): Length of the FFT window (default: 2048).
        hop_length (int): Hop length between frames (default: 512).
        n_mels (int): Number of mel bands to generate (default: 128).
    
    Returns:
        mel_spectrogram (ndarray): Mel spectrogram of the audio file.
        sr (int): Sampling rate used for the computation.
    """
    # Load audio file
    n_fft = int(sr * 0.025)
    hop_length = int(sr * 0.01)
    n_mels = 64
    center = False

    y, sr = librosa.load(audio_path, sr=sr)
    
    # Compute mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    return mel_spectrogram, sr

def plot_mel_spectrogram(mel_spectrogram, sr, figsize=(10, 4)):
    """
    Plot the mel spectrogram.
    
    Parameters:
        mel_spectrogram (ndarray): Mel spectrogram.
        sr (int): Sampling rate used for the computation.
    """
    hop_length = int(sr * 0.01)
    plt.figure(figsize=figsize)
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()

def frames_to_video(frames_folder, output_video_path, fps=10):
    """
    Convert individual frames saved as images to a video.
    
    Parameters:
        frames_folder (str): Path to the folder containing individual frame images.
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the output video (default: 10).
    """
    # Get list of image files in the frames folder
    frame_files = [f for f in os.listdir(frames_folder) if f.endswith('.png') or f.endswith('.jpg')]

    # Sort frame files by filename
    frame_files.sort()

    # Read first frame to get frame dimensions
    first_frame_path = os.path.join(frames_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write frames to video
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Release video writer
    video_writer.release()

    print(f"Video saved to {output_video_path}")

def get_frame_at_timestamp(video_path, timestamp):
    """
    Get the frame at a particular timestamp from a video file.
    
    Parameters:
        video_path (str): Path to the video file.
        timestamp (float): Timestamp in seconds.
    
    Returns:
        frame (ndarray): Frame at the specified timestamp, or None if timestamp is outside the video duration.
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get total number of frames and frame rate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print("Total frames: , length of video: ", total_frames, total_frames/fps)
    
    # Calculate frame index corresponding to the timestamp
    frame_index = int(timestamp * fps)
    
    # Check if frame index is within the valid range
    if frame_index < 0 or frame_index >= total_frames:
        print("Timestamp is outside the video duration.")
        return None
    
    # Seek to the frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # Read the frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    return frame
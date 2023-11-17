import numpy as np
import librosa
from pydub import AudioSegment
import cv2
from moviepy.editor import VideoFileClip
import os
# Step 1: Load the target sound effect and the audio or video file
target_sound_file = "./tt_samples/ping1.wav"
audio_or_video_file = "test.wav"
# Load the target sound effect
target_sound,sr = librosa.load(target_sound_file)
target_sound = np.array(target_sound)
# Load the audio or video file
audio, sr1 = librosa.load(audio_or_video_file)
length_video=int(len(audio)*1000/sr1)

segment_duration = int(len(target_sound)*1000/ sr)
# Step 2: Convert the target sound effect to a spectrogram# Step 3: Split the audio into short segments and compare them with the target sound effect  # Convert milliseconds to seconds)
segment_length = segment_duration
l=[]
samples=os.listdir("./tt_samples")
for i in range(0, length_video - segment_length,1000):
    l1=[]
    # Extract a segment from the audio
    segment = audio[i:i + segment_length]
    for j in range(len(samples)):
        sample=samples[j]
        audio_file=os.path.join('./tt_samples',sample)
        target_sound_spec = librosa.amplitude_to_db(np.abs(librosa.stft(target_sound, hop_length=512)), ref=np.max)

    # Convert the segment to a spectrogram using librosa
        segment_spec = librosa.amplitude_to_db(np.abs(librosa.stft(segment, hop_length=512)), ref=np.max)

    # Compare the spectrograms of the segment and target sound effect
        resized_target_sound_spec = np.resize(target_sound_spec, segment_spec.shape)

    # Compare the spectrograms of the segment and target sound effect
        similarity = np.mean(np.abs(segment_spec - resized_target_sound_spec))

    # Set a threshold to determine if the target sound effect is present
        threshold = 17  # Adjust this value based on your requirements

        if similarity < threshold:
            l1.append((similarity,j))
    if l1!=[]:
        a,b=max(l1)
        l.append((i/(1000),b))

print(l)
# Sound-Detection
The main objective of this project is to develop digital methods for detecting a reference sound in a video or audio file. We want to predict its starting time in the target sample.

## Code and details

* The notebook [1_introduction_to_sound_processing.ipynb](https://github.com/centralelyon/Sound-Detection/blob/main/1_introduction_to_sound_processing.ipynb) defines sound and introduces methods of sound processing including filtering and time stretching.
* The notebook [2_Feature_extraction_from_Audio_signal.ipynb](https://github.com/centralelyon/Sound-Detection/blob/main/2_Feature_extraction_from_Audio_signal.ipynb) explains the multiple ways of representing a sound signal and focuses on the information that we can extract from this signal : spectral_centroid_feature, spectral rolloff, spectral bandwidth, zero_crossing rate, MFCCs etc.
* The notebook [3_Sound_detection.ipynb](https://github.com/centralelyon/Sound-Detection/blob/main/3_Sound_detection.ipynb) defines our performance criteria , evaluate the different methods (feature_based, correlation_based, spectrogram_based)  on a labeled dataset (link to dataset : (https://drive.google.com/drive/folders/1LqGDj05BkmnG4FyfKOQcJo86juj4C5Sa?usp=sharing )) and finally  conclude on the best method to use.


## Results

<img src="https://github.com/centralelyon/Sound-Detection/blob/main/results.png" alt="Results" width="700"/>

The feature_extraction method is the most performant method. We can use it for further project that use sound detection.

Code of the method : 
```python
def generate_features(file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path)

    # Extract features
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    rms = librosa.feature.rms(y=audio)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
    # Concatenate the features into a single feature vector
    feature_vector = np.concatenate(
        (chroma_stft, spectral_centroid, spectral_bandwidth, spectral_rolloff, rms, zero_crossing_rate),
        axis=0
    )

    return feature_vector
    
def detect_sound_ref_feature_extraction(samples_path, audio_or_video_name, ref_sound_name,threshold=1000):
    # Load the target sound effect and the audio or video file
    target_sound_file = ref_sound_name
    audio_or_video_file = os.path.join(samples_path, audio_or_video_name)
    # Load the target sound effect
    target_sound, sr = librosa.load(target_sound_file)
    target_sound = np.array(target_sound)
    # Load the audio or video file
    audio, sr1 = librosa.load(audio_or_video_file)
    length_video = int(len(audio) * 1000 / sr1)

    segment_duration = int(len(target_sound) * 1000 / sr)
    # Convert the target sound effect to a spectrogram
    target_sound_feature = generate_features(target_sound_file)
    # Split the audio into short segments and compare them with the target sound effect  # Convert milliseconds to seconds)
    segment_length = segment_duration
    l = []
    for i in range(0, length_video - segment_length, segment_length):
            # Extract a segment from the audio
            segment = audio[i : i + segment_length]


            # Convert the segment to a spectrogram using librosa
            chroma_stft = librosa.feature.chroma_stft(y=segment, sr=sr1)
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr1)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr1)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr1)
            rms = librosa.feature.rms(y=segment)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)
    # Concatenate the features into a single feature vector
            segment_feature_vector = np.concatenate(
        (chroma_stft, spectral_centroid, spectral_bandwidth, spectral_rolloff, rms, zero_crossing_rate),
        axis=0
    )
            # Compare the features of the segment and target sound effect
            similarity = np.mean(np.abs(segment_feature_vector - target_sound_feature))
            print(similarity)

            # Set a threshold to determine if the target sound effect is present
              # Adjust this value based on your requirements

            if similarity < threshold:
                l.append(i / 1000)
    return l

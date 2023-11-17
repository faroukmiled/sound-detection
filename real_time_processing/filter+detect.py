from audio_buffer import AudioBuffer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib
import librosa,os
import time as tm
from scipy import signal
matplotlib.use('TkAgg')
important_freq=[1227, 5097, 10303, 16959, 20829, 1234, 6892, 10511, 15164, 20822, 639, 6957, 10554, 15099, 21418, 7373, 11028, 14683, 57, 4552, 9104, 12952, 17505, 21999, 5385, 8982, 13074, 16672, 5909, 9592, 16147, 5916, 10231, 16140, 10619, 6253, 9865, 15803, 4947, 8537, 13520, 17110, 1952, 7208, 14848, 20104, 2146, 7438, 14618, 19910, 1184, 6907, 11559, 15149, 20872, 5309, 9132, 12925, 16747, 1888, 7402, 14654, 20168, 5140, 8759, 13297, 16916, 1095, 6847, 10652, 15209, 20961]
def filter_frequencies(signal_data, sampling_rate, important_freq, bandwidth):
    # Normalize the important frequencies and bandwidth
    normalized_important_freq = np.array(important_freq) / (0.5 * sampling_rate)
    normalized_bandwidth = bandwidth / (0.5 * sampling_rate)

    # Design the band-pass filter
    b, a = signal.butter(4, [normalized_important_freq - normalized_bandwidth,
                             normalized_important_freq + normalized_bandwidth],
                         btype='band', analog=False)

    # Apply the filter to the signal
    filtered_signal = signal.lfilter(b, a, signal_data)

    return filtered_signal


def process_signal(indata):
    #processed_data=remove_noise(indata,44100,20000)
    processed_data=indata
    time_values=[]
    # Your processing code here
    #print("Processing signal:", indata)
    target_sound_file = "./tt_samples/ping11.wav"
    # Load the target sound effect
    target_sound,sr = librosa.load(target_sound_file)
    target_sound = np.array(target_sound)
    # Load the audio or video file
    length_video=len(indata)
    segment_duration = int(len(target_sound)*1000/ sr)
    # Step 2: Convert the target sound effect to a spectrogram# Step 3: Split the audio into short segments and compare them with the target sound effect  # Convert milliseconds to seconds)
    segment_length = segment_duration
    l=[]
    samples=os.listdir("./tt_samples")
    for i in range(0, length_video -segment_duration,segment_duration):
        l1=[]
        # Extract a segment from the audio
        segment = indata[i:i + segment_duration]

        for j in range(len(samples)):
            sample=samples[j]
            audio_file=os.path.join('./tt_samples',sample)
            target_sound,sr = librosa.load(audio_file)
            target_sound = np.array(target_sound)
            target_sound_spec = librosa.amplitude_to_db(np.abs(librosa.stft(target_sound, hop_length=512)), ref=np.max)

        # Convert the segment to a spectrogram using librosa
            segment_spec = librosa.amplitude_to_db(np.abs(librosa.stft(segment.astype(np.float32), hop_length=512)), ref=np.max)

        # Compare the spectrograms of the segment and target sound effect
            resized_target_sound_spec = np.resize(target_sound_spec, segment_spec.shape)

        # Compare the spectrograms of the segment and target sound effect
            similarity = np.mean(np.abs(segment_spec - resized_target_sound_spec))

        # Set a threshold to determine if the target sound effect is present
            threshold = 11 # Adjust this value based on your requirements

            if similarity < threshold:
                l1.append((similarity,j))
        if l1!=[]:
                time_values.append(i)
            
    return processed_data,time_values


if __name__ == "__main__":
    audio_buffer = AudioBuffer(chunks=10)
    audio_buffer.start()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    amp = 10000 # you might need to adjust this parameter
    #line, = ax.plot(amp * np.random.random(len(audio_buffer)) - amp/2)
    lines=ax.plot(amp * np.random.random(len(audio_buffer)) - amp/2)
    counter_text = ax.text(0.1, 0.9, "", transform=ax.transAxes)
    counter=0
    def animate(i):
        global counter
        data = audio_buffer()  # Get audio data from the buffer
        #data=filter_frequencies(data,44100,important_freq,0)

        # Process the data and generate time values
        processed_data, time_values = process_signal(data)

        # Update the plot based on the time values
        if time_values:
            for time in range(len(data)):
                if time in time_values:
                    counter+=1
                    # Highlight the plot in green

        for column, line in enumerate(lines): 
            line.set_ydata(data)
        counter_text.set_text(f"Counter: {counter}")
        return (lines+[counter_text])
    
    anim = animation.FuncAnimation(fig, animate, interval=1, blit=True)
    plt.show()
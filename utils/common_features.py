import librosa,os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
data_path="/home/fmiled/Bureau/compete/dataset/audio/fold1"
files=os.listdir(data_path)
l=[]
for file in files :
    sound_file=os.path.join(data_path,file)
    data,sample_rate=librosa.load(sound_file)
    time=librosa.times_like(data,sr=sample_rate)
    frequencies, times, spectrogram_data = spectrogram(data, fs=sample_rate)
    max_indices = np.argmax(spectrogram_data, axis=0)

    predominant_frequencies = frequencies[max_indices]
    l+=list(predominant_frequencies)
    #librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='log')
    #plt.show()
    """ plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data), shading='auto', cmap='inferno')
    plt.colorbar(label='Power Spectral Density (dB)')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # Plot the predominant frequencies
    plt.plot(times, predominant_frequencies, color='w', linewidth=2)
    plt.scatter(times, predominant_frequencies, color='w', s=10)

    plt.show() """
l=list(set(l))
video_path="./test/000007.mp4"
data,sample_rate=librosa.load(video_path)
frequencies, times, spectrogram_data = spectrogram(data, fs=sample_rate)
h,w=spectrogram_data.shape
count=[0]*w
for j in range(w):
    s=0
    for i in range(len(l)):
        if list(frequencies).count(l[i]):
            s+=1
    count[j]=s
print(count)
        




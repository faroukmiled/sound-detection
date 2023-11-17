import librosa
import os
import moviepy.editor as mp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, ifft, fftfreq
import scipy
data_path=os.listdir("./test")
times=[]
plots=[]
def temporal_plot(video_path_in,video_name)-> None:
    video_path=os.path.join(video_path_in,video_name)
    video = mp.VideoFileClip(video_path)
    fps=video.fps
    duration=video.duration
    audio_signal = video.audio.to_soundarray()[:, 0]
    time=librosa.times_like(audio_signal,sr=fps)
def frequency_analysis(video_path_in,video_name):
    video_path=os.path.join(video_path_in,video_name)
    audio, sr1 = librosa.load(video_path)
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft)
    freq = np.linspace(0, sr1, len(magnitude))
    peaks, _ = find_peaks(magnitude, distance=100)  # Adjust the distance parameter as needed
    important_freq = list(map(int,freq[peaks]))
    return important_freq
def filter_signal(f_min,f_max):
    data_path="./test"
    files=os.listdir(data_path)
    for file in (files):
        if file.endswith(".mp4"):
            audio, sr1 = librosa.load(os.path.join(data_path,file))
            signal= fft(audio)
            fs = 1000  # Sample rate (Hz)
            t = np.arange(0, 1, 1/fs) 
            freqs = fftfreq(len(t), 1/fs) 
            indices = np.where(np.logical_and(freqs >= f_min, freqs <= f_max))
            filtered_freq = signal.copy()
            filtered_freq[np.logical_not(np.isin(np.arange(len(freqs)), indices[0]))] = 0
            filtered_signal = ifft(filtered_freq)
            plt.plot(t,filtered_signal)
            plt.show()
        break
def convolve_signal():
    data_path="./test"
    files=os.listdir(data_path)
    for file in (files):
        if file.endswith(".mp4"):
            data,sample_rate = librosa.load(os.path.join(data_path,file))
            times = np.arange(len(data))/sample_rate
            # create a Hanning kernel 1/50th of a second wide
            kernel_width_seconds = 1.0/50
            kernel_size_points = int(kernel_width_seconds * sample_rate)
            kernel = np.hanning(kernel_size_points)
            # normalize the kernel
            kernel = kernel / kernel.sum()
            # Create a filtered signal by convolving the kernel with the original data
            filtered = np.convolve(kernel, data, mode='valid')
            plt.plot(np.arange(len(filtered))/sample_rate,filtered)
            plt.show()
def main():
    data_path="./test"
    files=os.listdir(data_path)
    freq_per_video=[]
    all_freq=[]
    for file in (files):
        if file.endswith(".mp4"):
            important_freq=frequency_analysis(data_path,file)
            freq_per_video.append(important_freq)
            all_freq+=important_freq
    all_freq=list(set(all_freq))
    d={}
    for freq in all_freq:
        d[freq]=0
    for frequencies in freq_per_video:
        for i in range(len(frequencies)):
            d[frequencies[i]]+=1
    x=sorted(list(d.keys()))
    y=[]
    for m in x:
        y.append(d[m])
    plt.scatter(x,y)
    plt.show()


if __name__=="__main__":
    convolve_signal()
"""
for file in (data_path):
    if file.endswith(".mp4"):
        video_path=os.path.join("./test",file)
        video = mp.VideoFileClip(video_path)
        fps=video.fps
        duration=video.duration
        audio_signal = video.audio.to_soundarray()[:, 0]
        time=librosa.times_like(audio_signal,sr=fps)
        times.append(time)
        plots.append(audio_signal)
for i,ax in enumerate(axs):
    ax.plot(times[i],plots[i])
#plt.tight_layout()
plt.show()
"""
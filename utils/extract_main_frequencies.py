import os
import librosa
from scipy.signal import find_peaks
import numpy as np
l=os.listdir("./tt_samples")
l1=[]
for file in l:
    file=os.path.join('./tt_samples',file)
    audio, sr = librosa.load(file)
    fft = np.fft.fft(audio)

        # Compute the magnitude spectrum
    magnitude = np.abs(fft)

    # Create the frequency axis
    freq = np.linspace(0, sr, len(magnitude))
    peaks, _ = find_peaks(magnitude, distance=500)  # Adjust the distance parameter as needed

    # Select the corresponding frequencies
    important_freq = freq[peaks]
    for i in range(len(important_freq)):
        if l1.count(int(important_freq[i]))==0:
            l1.append(int(important_freq[i]))
print(l1)
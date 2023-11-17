import numpy as np
from scipy.signal import butter,filtfilt# Filter requirements
import librosa
import plotly.graph_objects as go
from scipy.io import wavfile
# Filter requirements.
T = 5        # Sample Period
fs = 1000       # sample rate, Hz
cutoff = 20     # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hznyq = 0.5 * fs  # Nyquist Frequencyorder = 2       # sin wave can be approx represented as quadratic
nyq = 0.5 * fs  # Nyquist Frequency
order = 5       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples
sig,sample_rate=librosa.load("./test/000019.mp4")
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(sig, cutoff, sample_rate, order)

fig = go.Figure()
fig.add_trace(go.Scatter(
            y = sig,
            line =  dict(shape =  'spline' ),
            name = 'signal with noise'
            ))
fig.add_trace(go.Scatter(
            y = y,
            line =  dict(shape =  'spline' ),
            name = 'filtered signal'
            ))
fig.show()
wavfile.write("filtered_signal.wav", sample_rate,y)
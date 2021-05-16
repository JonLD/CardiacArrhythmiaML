import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import csv

ecg = scipy.io.loadmat('D:/Arythmia PPG-ECG/physio2017/training2017/A00003.mat')

sample_rate = 300
max_length = 30 * sample_rate
#let's plot the data we have
time = np.linspace(0, ecg['val'][0].size/sample_rate, ecg['val'][0].size)
ecg_fig, ecg_ax = plt.subplots()
ecg_ax.plot(time, ecg['val'][0])
ecg_ax.set_title("ECG Signal")
ecg_ax.set_ylabel('Amplitude (mV)')
ecg_ax.set_xlabel('time (s)')

minimum = np.min(ecg['val'][0])
maximum = np.max(ecg['val'][0])

ecg_signal_norm = (ecg['val'][0] - minimum) / (maximum - minimum)
ecg_fig_n, ecg_ax_n = plt.subplots()
ecg_ax_n.plot(time, ecg_signal_norm)
ecg_ax_n.set_title("ECG Signal Normalised")
ecg_ax_n.set_ylabel('Amplitude')
ecg_ax_n.set_xlabel('time (s)')

if(ecg['val'][0].size < max_length):
    ecg_seq = np.array(ecg['val'][0])
    ecg_seq = np.pad(ecg_seq, (0, max_length - ecg['val'][0].size), 'constant')

if(ecg['val'][0].size > max_length):
    ecg_seq = np.array(ecg['val'][0])
    ecq_split = []
    if((ecg['val'][0].size / max_length) > 2):
        ecq_split = np.split(ecg_seq, 3)
    else:
        ecq_split = np.split(ecg_seq, 2)
    for i, array in enumerate(ecq_split):
        ecq_split[i] = np.pad(array, (0, max_length - array.size), 'constant')
    
if(ecg['val'][0].size == max_length):
    pass


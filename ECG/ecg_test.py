import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import csv

def plot(name, x_name, y_name, signal, x_range):
    no_bw_fig, no_bw_ax = plt.subplots()
    no_bw_ax.plot(x_range, signal)
    no_bw_ax.set_title(name)
    no_bw_ax.set_ylabel(y_name)
    no_bw_ax.set_xlabel(x_name)
    return no_bw_fig
#        fig1 = plot('ECG Signal', 'Time (s)', 'Amplitude (mV)', ecg_array, time)
#        fig2 = plot('ECG Signal Normalised', 'Time (s)', 'Amplitude', ecg_signal_norm, time)
#        fig1.savefig(os.path.join(target, filename[:-4] + '.png'))
#        fig2.savefig(os.path.join(target, filename[:-4] + '_normalised.png'))



ecg = scipy.io.loadmat('D:/Arythmia PPG-ECG/physio2017/training2017/A00003.mat')
cvs = 'D:/Arythmia PPG-ECG/physio2017/training2017/REFERENCE-v3.csv'
sample_rate = 300
max_length = 30 * sample_rate
#let's plot the data we have
time = np.linspace(0, ecg['val'][0].size/sample_rate, ecg['val'][0].size)
ecg_fig, ecg_ax = plt.subplots()
ecg_ax.plot(time, ecg['val'][0])
ecg_ax.set_title('ECG Signal')
ecg_ax.set_ylabel('Amplitude (mV)')
ecg_ax.set_xlabel('time (s)')

minimum = np.min(ecg['val'][0])
maximum = np.max(ecg['val'][0])

ecg_signal_norm = (ecg['val'][0] - minimum) / (maximum - minimum)
ecg_fig_n, ecg_ax_n = plt.subplots()
ecg_ax_n.plot(time, ecg_signal_norm)
ecg_ax_n.set_title('ECG Signal Normalised')
ecg_ax_n.set_ylabel('Amplitude')
ecg_ax_n.set_xlabel('time (s)')

#just for padding the ecgs if they shorter than 30s
#if they longer then splitting them and padding them
#plus adding a padded label to the padded ones
#reading out the labels from the csv file
labels = {}
with open(cvs, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        labels[row[0][4:-2]] = row[0][-1:]
print(labels)
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
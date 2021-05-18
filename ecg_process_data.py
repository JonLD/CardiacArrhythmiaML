from os import listdir
from os.path import isfile, join
import os
from pathlib import Path
import csv
import scipy.io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

target_dir = "D:/Arythmia PPG-ECG/processed_data/physio2017"
sample_rate = 300
max_length = 30 * sample_rate

#function needed when the split would produce unequal subarrays and throw an error
def split_padded(a,n):
    padding = (-len(a))%n
    pad_array = np.zeros(n)
    if(padding > 0):
        pad_array[n-1] = 1
    return np.split(np.concatenate((a,np.zeros(padding))),n), pad_array

#just making sure the split is done corrrectly depending the size
#and adding the padded tag if it is padded
def SplitEcg(ecg):
    ecg_seq = np.array(ecg)
    if(ecg.size < max_length):
        ecg_seq = np.pad(ecg_seq, (0, max_length - ecg.size), 'constant')
        pad_array = np.array([1])
        return np.array(ecg_seq), pad_array

    if(ecg.size > max_length):
        ecq_split =  np.array([])
        pad_array = np.array([])
        if((ecg.size / max_length) > 2): 
            ecq_split, pad_array = split_padded(ecg_seq, 3)
        else:
            ecq_split, pad_array = split_padded(ecg_seq, 2)
        for i, array in enumerate(ecq_split):
            if((max_length - array.size) > 0):
                pad_array[i] = 1
            else:
                pad_array[i] = 0
            ecq_split[i] = np.pad(array, (0, max_length - array.size), 'constant')
        return np.array(ecq_split), pad_array
    if(ecg.size == max_length):
        pad_array = np.zeros(1)
        return np.array(ecg), pad_array
def plot(name, x_name, y_name, signal, x_range):
    no_bw_fig, no_bw_ax = plt.subplots()
    no_bw_ax.plot(x_range, signal)
    no_bw_ax.set_title(name)
    no_bw_ax.set_ylabel(y_name)
    no_bw_ax.set_xlabel(x_name)
    return no_bw_fig

def ProcessData(filename, directory, target, label):
    target = os.path.join(target, filename[:-4])
    Path(target).mkdir(parents=True, exist_ok=True)
    ecg = scipy.io.loadmat(os.path.join(directory, filename))
    ecg_array, pad_array = SplitEcg(ecg['val'][0])
    minimum = np.min(ecg['val'][0])
    maximum = np.max(ecg['val'][0])
    class_labels = ['N','A','O','~']
    time = np.linspace(0, 30, 9000)
    if(ecg_array.ndim == 1):
        class_array = [0,0,0,0,0]
        ecg_signal_norm = (ecg_array - minimum) / (maximum - minimum)
        for i, class_name in enumerate(class_labels):
            if(label == class_name):
                class_array[i] = 1
        class_array[4] = pad_array[0]
        np.save(os.path.join(target, filename[:-4]), ecg_array)
        np.save(os.path.join(target, filename[:-4] + '_normalised'), ecg_signal_norm)
        np.save(os.path.join(target, filename[:-4] + '_target'), class_array)
        fig1 = plot('ECG Signal', 'Time (s)', 'Amplitude (mV)', ecg_array, time)
        fig2 = plot('ECG Signal Normalised', 'Time (s)', 'Amplitude', ecg_signal_norm, time)
        fig1.savefig(os.path.join(target, filename[:-4] + '.png'))
        fig2.savefig(os.path.join(target, filename[:-4] + '_normalised.png'))
    else:
        for i, ecg_sig in enumerate(ecg_array):
            class_array = [0,0,0,0,0]
            ecg_signal_norm = (ecg_sig - minimum) / (maximum - minimum)
            for k, class_name in enumerate(class_labels):
                if(label == class_name):
                    class_array[k] = 1
            class_array[4] = pad_array[i - 1]
            np.save(os.path.join(target, filename[:-4] + str(i)), ecg_sig)
            np.save(os.path.join(target, filename[:-4] + '_normalised' + str(i)), ecg_signal_norm)
            np.save(os.path.join(target, filename[:-4] + '_target' + str(i)), class_array)
            fig1 = plot('ECG Signal', 'Time (s)', 'Amplitude (mV)', ecg_sig, time)
            fig2 = plot('ECG Signal Normalised', 'Time (s)', 'Amplitude', ecg_signal_norm, time)
            fig1.savefig(os.path.join(target, filename[:-4] + str(i) + '.png'))
            fig2.savefig(os.path.join(target, filename[:-4] + str(i) + '_normalised.png'))
        
    
    

def GetLabels(file_path):
    labels = {}
    with open(file_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            labels[row[0][4:-2]] = row[0][-1:]
    return labels

physio2017_train = "D:/Arythmia PPG-ECG/physio2017/training2017"

files = [f for f in listdir(physio2017_train) if isfile(join(physio2017_train, f)) and not f.endswith('.hea')]
train_data = files[:-4]
labels = files[-2]

train_cvs = os.path.join(physio2017_train, labels)
train_labels = GetLabels(train_cvs)

for i,file in enumerate(train_data):
    ProcessData(file, physio2017_train, target_dir, train_labels[file[:-4]])
from os import listdir
from os.path import isfile, join
import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import pywt
from pathlib import Path

target_dir = "D:/Arythmia PPG-ECG/processed_data/brno"
#well brno_test is basically this just for one file
def ProcessData(filenames, directory, target):
    target = os.path.join(target, filenames[0][:-4])
    Path(target).mkdir(parents=True, exist_ok=True)
    ecg_record = wfdb.rdrecord(os.path.join(directory, filenames[0]))
    ppg_record = wfdb.rdrecord(os.path.join(directory, filenames[1]))
    ecg_dictionary = ecg_record.__dict__
    ppg_dictionary = ppg_record.__dict__
    
    ecg_signal = ecg_dictionary['p_signal'][0]
    time = np.arange(ecg_signal.size)

    ecg_fig, ecg_ax = plt.subplots()
    ecg_ax.plot(time, ecg_signal)
    ecg_ax.set_title("ECG Signal")
    ecg_ax.set_ylabel('uV')
    ecg_ax.set_xlabel('time (ms)')
    ecg_fig.savefig(os.path.join(target, filenames[0] + '.png'))
    
    ppg_signal = ppg_dictionary['p_signal'][0]
    time2 = np.arange(ppg_signal.size)

    ppg_fig, ppg_ax = plt.subplots()
    ppg_ax.plot(time2, ppg_signal)
    ppg_ax.set_title("PPG Signal")
    ppg_ax.set_ylabel('Amplitude')
    ppg_ax.set_xlabel('time (s/30)')
    ppg_fig.savefig(os.path.join(target, filenames[1] + '.png'))
    
    coeffs = pywt.wavedec(ecg_signal, 'db8', level=8)
    dwt_fig, dwt_axs = plt.subplots(9)
    dwt_fig.suptitle('ECG signal decomposition')
    for i in range(9):
        dwt_axs[i].plot(np.arange(coeffs[i].size), coeffs[i])
    coeffs[0] = np.zeros_like(coeffs[0])
    no_bw_signal = pywt.waverec(coeffs, 'db8')
    no_bw_fig, no_bw_ax = plt.subplots()
    no_bw_ax.plot(time, no_bw_signal)
    no_bw_ax.set_title("ECG Signal")
    no_bw_ax.set_ylabel('uV')
    no_bw_ax.set_xlabel('time (ms)')
    dwt_fig.savefig(os.path.join(target, filenames[0] + 'decomposition.png'))
    no_bw_fig.savefig(os.path.join(target, filenames[0] + 'corrected.png'))
    np.save(os.path.join(target, filenames[0]), no_bw_signal)
    np.save(os.path.join(target, filenames[1]), ppg_signal)

    coeffs2 = pywt.wavedec(ppg_signal, 'db8', level=8)
    dwt2_fig, dwt2_axs = plt.subplots(9)
    dwt2_fig.suptitle('PPG signal decomposition')
    for i in range(9):
        dwt2_axs[i].plot(np.arange(coeffs2[i].size), coeffs2[i])
    dwt2_fig.savefig(os.path.join(target, filenames[1] + 'decomposition.png'))

        

brno = "D:/Arythmia PPG-ECG/brno/brno-university-of-technology-smartphone-ppg-database-but-ppg-1.0.0"

directory = [x[0] for x in os.walk(brno)]
directory = directory[1::]
#just looping the directories and reading the files not that intresting
#probably map function would be faster than the native python loop
#running this can take a while
for i in range(len(directory)):
    files = [f for f in listdir(directory[i]) if isfile(join(directory[i], f))]
    file_names = [files[0][:-4], files[2][:-4]]
    ProcessData(file_names, directory[i], target_dir)
    


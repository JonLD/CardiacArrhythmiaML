import scipy.io
import numpy as np
import matplotlib.pyplot as plt


mat = scipy.io.loadmat('D:/Arythmia PPG-ECG/data_IEEE2015/TrueBPM/True_S01_T01.mat')
mat2 = scipy.io.loadmat('D:/Arythmia PPG-ECG/data_IEEE2015/Training_data/DATA_01_TYPE01.mat')
mat3 = scipy.io.loadmat('D:/Arythmia PPG-ECG/data_IEEE2015/Training_data/DATA_01_TYPE01_BPMtrace.mat')
mat4 = scipy.io.loadmat('D:/Arythmia PPG-ECG/data_IEEE2015/TestData/TEST_S01_T01')
mat5 = scipy.io.loadmat('D:/Arythmia PPG-ECG/data_IEEE2015/Extra_TrainingData/BPM_S04_T01.mat')
mat6 = scipy.io.loadmat('D:/Arythmia PPG-ECG/data_IEEE2015/Extra_TrainingData/DATA_S04_T01.mat')

#let's plot the data we have for the True BPM and the Test Data as they are related
time = np.arange(mat['BPM0'].size)
bpm_fig, bpm_ax = plt.subplots()
bpm_ax.plot(time, mat['BPM0'])
bpm_ax.set_title("BPM Signal")
bpm_ax.set_ylabel('uV')
bpm_ax.set_xlabel('time (ms)')

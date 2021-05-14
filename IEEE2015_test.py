1import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pywt

BPM_test = scipy.io.loadmat('D:/Arythmia PPG-ECG/data_IEEE2015/TrueBPM/True_S01_T01.mat')
PPG_test = scipy.io.loadmat('D:/Arythmia PPG-ECG/data_IEEE2015/TestData/TEST_S01_T01')
PPG_train = scipy.io.loadmat('D:/Arythmia PPG-ECG/data_IEEE2015/Training_data/DATA_01_TYPE01.mat')
BPM_train = scipy.io.loadmat('D:/Arythmia PPG-ECG/data_IEEE2015/Training_data/DATA_01_TYPE01_BPMtrace.mat')
BPM_train_extra = scipy.io.loadmat('D:/Arythmia PPG-ECG/data_IEEE2015/Extra_TrainingData/BPM_S04_T01.mat')
PPG_train_extra = scipy.io.loadmat('D:/Arythmia PPG-ECG/data_IEEE2015/Extra_TrainingData/DATA_S04_T01.mat')

#let's plot the data we have for the True BPM and the Test Data as they are related
time = np.arange(BPM_test['BPM0'].size)
bpm_fig, bpm_ax = plt.subplots()
bpm_ax.plot(time, BPM_test['BPM0'])
bpm_ax.set_title("BPM Signal")
bpm_ax.set_ylabel('Beats per minute')
bpm_ax.set_xlabel('time (ms)')

#test data first two row is ppg and last three is tri-acceleration signals
ppg1 = PPG_test['sig'][0]
ppg2 = PPG_test['sig'][1]
gyro_x = PPG_test['sig'][2]
gyro_y = PPG_test['sig'][3]
gyro_z = PPG_test['sig'][4]
time2 = np.arange(ppg1.size)

ppg1_fig, ppg1_ax = plt.subplots()
ppg1_ax.plot(time2, ppg1)
ppg1_ax.set_title("PPG Signal")
ppg1_ax.set_ylabel('uV')
ppg1_ax.set_xlabel('time (ms)')

ppg2_fig, ppg2_ax = plt.subplots()
ppg2_ax.plot(time2, ppg2)
ppg2_ax.set_title("PPG-2 Signal")
ppg2_ax.set_ylabel('uV')
ppg2_ax.set_xlabel('time (ms)')

gyro_x_fig, gyro_x_ax = plt.subplots()
gyro_x_ax.plot(time2, gyro_x)
gyro_x_ax.set_title("Gyro_x Signal")
gyro_x_ax.set_ylabel('uV')
gyro_x_ax.set_xlabel('time (ms)')

gyro_y_fig, gyro_y_ax = plt.subplots()
gyro_y_ax.plot(time2, gyro_y)
gyro_y_ax.set_title("Gyro_y Signal")
gyro_y_ax.set_ylabel('uV')
gyro_y_ax.set_xlabel('time (ms)')

gyro_z_fig, gyro_z_ax = plt.subplots()
gyro_z_ax.plot(time2, gyro_z)
gyro_z_ax.set_title("Gyro_z Signal")
gyro_z_ax.set_ylabel('uV')
gyro_z_ax.set_xlabel('time (ms)')


#most above is not that inresting for us but I have done it for the sake if
#someone is intrested in using that data for something it does not have
#an ecg signal paired so for our project we cannot use them although subject 8
#containst valuable data, abnormal heart rhythm and blood pressure
ecg_train = PPG_train['sig'][0]
ppg_train1 = PPG_train['sig'][0]
ppg_train2 = PPG_train['sig'][0]
time2 = np.arange(ecg_train.size)

ecg_train_fig, ecg_train_ax = plt.subplots()
ecg_train_ax.plot(time2, ecg_train)
ecg_train_ax.set_title("ECG Signal")
ecg_train_ax.set_ylabel('uV')
ecg_train_ax.set_xlabel('time (ms)')

#removing baseline drift in ECG signal
#could use empirical mode decomposition but I used a discreet wavelet transform
#https://link.springer.com/content/pdf/10.1007%2F978-3-642-36321-4_47.pdf
#https://core.ac.uk/download/pdf/194317653.pdf
#https://www.hindawi.com/journals/jam/2013/763903/
coeffs = pywt.wavedec(ecg_train, 'db9', level=9)
dwt_fig, dwt_axs = plt.subplots(10)
dwt_fig.suptitle('ECG signal decomposition')
for i in range(10):
    dwt_axs[i].plot(np.arange(coeffs[i].size), coeffs[i])

#if you look at the plot the first decomposition looks like the baseline wander
#so we just zero it out
#and the second looks like BW too so I just zero it
coeffs[0] = np.zeros_like(coeffs[0])
coeffs[1] = np.zeros_like(coeffs[1])
no_bw_signal = pywt.waverec(coeffs, 'db9')
time2 = np.arange(no_bw_signal.size) #somehow it is 1 datapoint less after the transfrom
no_bw_fig, no_bw_ax = plt.subplots()
no_bw_ax.plot(time2, no_bw_signal)
no_bw_ax.set_title("ECG Signal")
no_bw_ax.set_ylabel('uV')
no_bw_ax.set_xlabel('time (ms)')

time2 = np.arange(ppg_train1.size)
ppg_train_fig, ppg_train_ax = plt.subplots()
ppg_train_ax.plot(time2, ppg_train1)
ppg_train_ax.set_title("PPG Signal 1")
ppg_train_ax.set_ylabel('uV')
ppg_train_ax.set_xlabel('time (ms)')

ppg_train_fig2, ppg_train_ax2 = plt.subplots()
ppg_train_ax2.plot(time2, ppg_train2)
ppg_train_ax2.set_title("PPG Signal 1")
ppg_train_ax2.set_ylabel('uV')
ppg_train_ax2.set_xlabel('time (ms)')
#

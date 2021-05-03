import matplotlib.pyplot as plt
import numpy as np
import wfdb
import pywt


#opening the ECG and PPG records
ecg_record = wfdb.rdrecord('D:/Arythmia PPG-ECG/brno/brno-university-of-technology-smartphone-ppg-database-but-ppg-1.0.0/100001/100001_ECG')

ppg_record = wfdb.rdrecord('D:/Arythmia PPG-ECG/brno/brno-university-of-technology-smartphone-ppg-database-but-ppg-1.0.0/100001/100001_PPG')

#just getting the dictionary out easier to work with in python
ecg_dictionary = ecg_record.__dict__
ppg_dictionary = ppg_record.__dict__


#just looking up the keys you might want to print out the original dictionary
#alternatively if you use spyder you can just check it from the variable explorer
print(ecg_dictionary.keys())

print(ecg_dictionary['record_name'])

#just getting the ecg signal and plotting the ecg
ecg_signal = ecg_dictionary['p_signal'][0]
time = np.arange(ecg_signal.size)

ecg_fig, ecg_ax = plt.subplots()
ecg_ax.plot(time, ecg_signal)
ecg_ax.set_title("ECG Signal")
ecg_ax.set_ylabel('uV')
ecg_ax.set_xlabel('time (ms)')

#same as above just from the ppg

ppg_signal = ppg_dictionary['p_signal'][0]
time2 = np.arange(ppg_signal.size)

ppg_fig, ppg_ax = plt.subplots()
ppg_ax.plot(time2, ppg_signal)
ppg_ax.set_title("PPG Signal")
ppg_ax.set_ylabel('Amplitude')
ppg_ax.set_xlabel('time (s/30)')

#removing baseline drift in ECG signal
#could use empirical mode decomposition but I used a discreet wavelet transform
#https://link.springer.com/content/pdf/10.1007%2F978-3-642-36321-4_47.pdf
#https://core.ac.uk/download/pdf/194317653.pdf
#https://www.hindawi.com/journals/jam/2013/763903/
coeffs = pywt.wavedec(ecg_signal, 'db8', level=8)
dwt_fig, dwt_axs = plt.subplots(9)
dwt_fig.suptitle('ECG signal decomposition')
for i in range(9):
    dwt_axs[i].plot(np.arange(coeffs[i].size), coeffs[i])

#if you look at the plot the first decomposition looks like the baseline wander
#so we just zero it out
coeffs[0] = np.zeros_like(coeffs[0])
no_bw_signal = pywt.waverec(coeffs, 'db8')
no_bw_fig, no_bw_ax = plt.subplots()
no_bw_ax.plot(time, no_bw_signal)
no_bw_ax.set_title("ECG Signal")
no_bw_ax.set_ylabel('uV')
no_bw_ax.set_xlabel('time (ms)')

coeffs2 = pywt.wavedec(ppg_signal, 'db8', level=8)
dwt2_fig, dwt2_axs = plt.subplots(9)
dwt2_fig.suptitle('PPG signal decomposition')
for i in range(9):
    dwt2_axs[i].plot(np.arange(coeffs2[i].size), coeffs2[i])

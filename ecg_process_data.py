import scipy.io
import numpy as np
import glob
import csv
from tqdm import tqdm
from sklearn import preprocessing

# Parameters
physio2017_train = "D:/Arythmia PPG-ECG/physio2017/training2017/"
target_dir = "D:/Arythmia PPG-ECG/processed_data/physio2017"

sample_rate = 300
max_length = 60*sample_rate


# Loading ECG data (.mat files)
files = sorted(glob.glob(physio2017_train+"*.mat"))
trainset = np.zeros((len(files),max_length))
trainset_normalised = np.zeros((len(files),max_length))
for i, file in enumerate(tqdm(files)):
    file_name = file[-10:-4]
    ecg_data = scipy.io.loadmat(file[:-4] + ".mat")
    data = ecg_data['val'].squeeze()
    data = np.nan_to_num(data) # removing NaNs and Infs
    minimum = np.min(data)
    maximum = np.max(data)
    data_normalise = (data - minimum) / (maximum - minimum)
    #padding sequence
    trainset[i,:min(max_length,len(data))] = data[:min(max_length,len(data))].T
    trainset_normalised[i,:min(max_length,len(data))] = data_normalise[:min(max_length,len(data))].T
    
# Loading labels
csvfile = list(csv.reader(open(physio2017_train+'REFERENCE-v3.csv')))
traintarget = np.zeros((trainset.shape[0],4))
classes = ['A','N','O','~']
for row in range(len(csvfile)):
    traintarget[row,classes.index(csvfile[row][1])] = 1
            
# Saving the labels and the data
scipy.io.savemat('trainingset.mat',mdict={'trainset': trainset,'traintarget': traintarget})
scipy.io.savemat('trainingset_normalised.mat',mdict={'trainset': trainset_normalised,'traintarget': traintarget})

    
    
    
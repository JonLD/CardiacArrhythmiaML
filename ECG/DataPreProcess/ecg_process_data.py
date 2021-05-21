import scipy.io
import numpy as np
import glob
import csv
from tqdm import tqdm

# Parameters
physio2017_train = "D:/Arythmia PPG-ECG/physio2017/training2017/"
physio2017_test = "D:/Arythmia PPG-ECG/physio2017/sample2017/validation/"

sample_rate = 300
max_length = 60*sample_rate


# Loading ECG data (.mat files)
def process_data(directory, target_file_name):
    files = sorted(glob.glob(directory+"*.mat"))
    dataset = np.zeros((len(files),max_length))
    dataset_normalised = np.zeros((len(files),max_length))
    for i, file in enumerate(tqdm(files)):
        ecg_data = scipy.io.loadmat(file[:-4] + ".mat")
        data = ecg_data['val'].squeeze()
        data = np.nan_to_num(data) # removing NaNs and Infs
        minimum = np.min(data)
        maximum = np.max(data)
        data_normalise = (data - minimum) / (maximum - minimum)
        #padding sequence
        dataset[i,:min(max_length,len(data))] = data[:min(max_length,len(data))].T
        dataset_normalised[i,:min(max_length,len(data))] = data_normalise[:min(max_length,len(data))].T
        
    # Loading labels
    csvfile = list(csv.reader(open(directory+'REFERENCE-v3.csv')))
    target = np.zeros((dataset.shape[0],4))
    classes = ['A','N','O','~']
    for row in range(len(csvfile)):
        target[row,classes.index(csvfile[row][1])] = 1
                
    # Saving the labels and the data
    scipy.io.savemat(target_file_name + '.mat',mdict={'data': dataset, 'target': target})
    scipy.io.savemat(target_file_name + '_normalised.mat',mdict={'data': dataset_normalised, 'target': target})

process_data(physio2017_train, 'trainingset')
process_data(physio2017_test, 'testset')
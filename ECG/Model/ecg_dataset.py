import scipy.io
import os.path
import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    training_file = 'trainingset.mat'
    training_file_normalsied = 'trainingset_normalised.mat'
    test_file = 'testset.mat'
    test_file_normalsied = 'testset_normalised.mat'
    def __init__(
            self,
            folder,
            sample_rate,
            time_length,
            train: bool = True,
            normal: bool = True):
        self.train = train
        self.normal = normal
        self.seq_length = time_length*sample_rate
        if self.train:
            if self.normal:
                data_file = self.training_file_normalsied
            else:
                data_file = self.training_file
        else:
            if self.normal:
                data_file = self.test_file_normalsied
            else:
                data_file = self.test_file
        self.data = self.loaddata(os.path.join(folder, data_file))
        

    def __len__(self):
        return self.data[0].shape[0]
    def __getitem__(self, index):
        X = self.data[0][index]
        Y = self.data[1][index]
        return X, Y
    def loaddata(self, data):    
#        Load training/test data into workspace        
#        This function assumes you have downloaded and padded/truncated the 
#        training set into a local file with ecg_process_data.py. This file should 
#        contain the following structures:
#            - trainset: NxM matrix of N ECG segments with length M
#            - traintarget: Nx4 matrix of coded labels where each column contains
#            one in case it matches ['A', 'N', 'O', '~'].
             
        matfile = scipy.io.loadmat(data)
        X = matfile['data']
        Y = matfile['target']

        X =  X[:,0:self.seq_length]
        X = torch.tensor(X)
        X = X.reshape(X.shape[0], 1, self.seq_length)
        Y = torch.tensor(Y)
        Y = Y.reshape(Y.shape[0], 4)
        Y = Y.int()
        if(self.normal):
            X = X.float()
        return (X, Y)
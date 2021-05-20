import scipy.io
from torch.utils.data import Dataset, DataLoader

class ECGDataset(Dataset):
    def __init__(self, data, sample_rate, time_length):
        self.seq_length = time_length*sample_rate
        self.data = self.loaddata(data)
        

    def __len__(self):
        return self.data[0].shape[0]
    def __getitem__(self, index):
        X = self.data[0][index]
        Y = self.data[1][index]
        return X, Y
    def loaddata(self, data):    
#        Load training/test data into workspace        
#        This function assumes you have downloaded and padded/truncated the 
#        training set into a local file named "trainingset.mat". This file should 
#        contain the following structures:
#            - trainset: NxM matrix of N ECG segments with length M
#            - traintarget: Nx4 matrix of coded labels where each column contains
#            one in case it matches ['A', 'N', 'O', '~'].
             
        matfile = scipy.io.loadmat(data)
        X = matfile['trainset']
        Y = matfile['traintarget']

        X =  X[:,0:self.seq_length] 
        return (X, Y)
    
training_data = ECGDataset('trainingset.mat', 300, 30)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
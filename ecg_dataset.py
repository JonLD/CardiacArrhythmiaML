from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, data, seq_length, target_length):
        self.data = data
        self.seq_length = seq_length
        self.target_length = target_length
        

    def __len__(self):
        return len(self.data.index)


    def __getitem__(self, index):
        row =  self.data.loc[[index]]
        X = row.values[0][1:-self.target_length]
        Y = row.values[0][-self.target_length:].astype(int)
        return X, Y


df = pd.read_csv('D:/Arythmia PPG-ECG/processed_data/physio2017train.csv')

training_data = ECGDataset(df, 9000, 5)
#train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
#train_features, train_labels = next(iter(train_dataloader))
#print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")
#implement oversampling
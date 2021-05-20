from torch.utils.data import DataLoader
from ecg_dataset import ECGDataset
import pytorch_lightning as pl

class ECGDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            ecg_full = ECGDataset(self.data_dir, train=True, normal=True)
            self.ecg_train, self.ecg_val = random_split(ecg_full, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.ecg_test = ECGDataset(self.data_dir, train=False, normal=True)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.ecg_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.ecg_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.ecg_test, batch_size=32)
    
training_data = ECGDataset('trainingset.mat', 300, 30)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
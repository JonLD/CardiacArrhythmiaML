import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from ecg_dataset import ECGDataset

class ECGDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_dir: str = './',
            val_split: int = 1279,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            normalised: bool = True,
            *args,
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.seed = seed
        self.val_split = val_split
        self.normalised = normalised
        self.batch_size = batch_size

    def train_dataloader(self):
        ecg_full = ECGDataset(self.data_dir, 300, 30, train=True, normal=self.normalised)
        train_length = len(ecg_full)
        ecg_train, _ = random_split(
            ecg_full,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            ecg_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        ecg_full = ECGDataset(self.data_dir, 300, 30, train=True, normal=self.normalised)
        train_length = len(ecg_full)
        _, ecg_val = random_split(
            ecg_full,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            ecg_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def test_dataloader(self):
        ecg_full = ECGDataset(self.data_dir, 300, 30, train=True, normal=self.normalised)
        loader = DataLoader(
            ecg_full,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy
import torch

class MLECG(LightningModule):
    def __init__(self, cell_count, batch_size, config):
        super().__init__()
        self.target_class = 4
        self.kernel_size = 11
        self.momentum = 0.99
        self.epsilon = 0.001
        self.dropout = 0.1
        self.batch_size = batch_size
        self.lr = config["lr"]
        
        self.layer1 = nn.Conv1d(1, 1, self.kernel_size)
        self.bnorm1 = nn.BatchNorm1d(1, self.epsilon, self.momentum, affine=False)
        self.maxp =  nn.MaxPool1d(10)
        self.rnn = nn.LSTM(1, self.target_class, cell_count, dropout=self.dropout, batch_first=True)
        self.dense = nn.Linear(899, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.bnorm1(x)
        x = self.maxp(x)
        x = x.reshape(self.batch_size, 899, 1)
        x, _ = self.rnn(x)
        x = x.reshape(self.batch_size, 4, 899)
        x = self.dense(x)
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch_data, batch_index):
        x, y = batch_data
        logits = self(x)
        loss = nn.BCEWithLogitsLoss(logits, y)
        
        # training metrics
        acc = accuracy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', torch.tensor(acc), on_step=True, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, batch_data, batch_index):
        x, y = batch_data
        logits = self(x)
        loss = nn.BCEWithLogitsLoss(logits, y)

        # validation metrics
        acc = accuracy(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', torch.tensor(acc), prog_bar=True)
        return {"val_loss": loss, "val_accuracy": acc}
    
    def test_step(self, batch_data, batch_index):
        x, y = batch_data
        logits = self(x)
        loss = nn.BCEWithLogitsLoss(logits, y)
        
        # validation metrics
        acc = accuracy(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack(
            [x["val_accuracy"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        self.log("avg_/val_accuracy", avg_acc)
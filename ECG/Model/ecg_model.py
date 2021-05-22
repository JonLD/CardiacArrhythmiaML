import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy

class MLECG(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.target_class = 4
        self.kernel_size = 11
        self.momentum = 0.99
        self.epsilon = 0.001
        self.dropout = 0.1
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.cell_count = config["lstm_size"]
        
        self.layer1 = nn.Conv1d(1, 1, self.kernel_size)
        self.bnorm1 = nn.BatchNorm1d(1, self.epsilon, self.momentum, affine=False)
        self.maxp =  nn.MaxPool1d(10)
        self.rnn = nn.LSTM(1, self.target_class, self.cell_count, dropout=self.dropout, batch_first=True)
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
        x = x.reshape(self.batch_size, 4)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch_data, batch_index):
        x, y = batch_data
        logits = self(x)
        criterion = nn.CrossEntropyLoss()
        probs = torch.softmax(logits, dim=1)
        
        # validation metrics
        acc = accuracy(torch.argmax(probs, dim=1), torch.argmax(y, dim=1))
        loss = criterion(logits, torch.argmax(y, dim=1))
        self.log('loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss, "train_accuracy": acc}
    
    def validation_step(self, batch_data, batch_index):
        x, y = batch_data
        logits = self(x)
        criterion = nn.CrossEntropyLoss()
        probs = torch.softmax(logits, dim=1)
        
        # validation metrics
        acc = accuracy(torch.argmax(probs, dim=1), torch.argmax(y, dim=1))
        loss = criterion(logits, torch.argmax(y, dim=1))
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', acc, prog_bar=True)
        return {"val_loss": loss, "val_accuracy": acc}
    
    def test_step(self, batch_data, batch_index):
        x, y = batch_data
        logits = self(x)
        criterion = nn.CrossEntropyLoss()
        probs = torch.softmax(logits, dim=1)
        
        # validation metrics
        acc = accuracy(torch.argmax(probs, dim=1), torch.argmax(y, dim=1))
        loss = criterion(torch.argmax(logits), torch.argmax(y, dim=1))
        self.log('test/loss', loss, prog_bar=True)
        self.log('test/accuracy', acc, prog_bar=True)
        return {"test_loss": loss, "test_accuracy": acc}
    
    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack(
            [x["loss"] for x in training_step_outputs]).mean()
        avg_acc = torch.stack(
            [x["train_accuracy"] for x in training_step_outputs]).mean()
        self.log("ptl/train_loss", avg_loss)
        self.log("ptl/train_accuracy", avg_acc)
    
    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack(
            [x["val_loss"] for x in validation_step_outputs]).mean()
        avg_acc = torch.stack(
            [x["val_accuracy"] for x in validation_step_outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)
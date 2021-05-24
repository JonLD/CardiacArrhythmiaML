import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy, f1


class MLECG(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.target_class = 4
        self.kernel_size = 11
        self.dropout = 0.1
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.cell_count = config["lstm_size"]
        
        self.layer1 = nn.Conv1d(1, 16, self.kernel_size)
        self.bnorm1 = nn.BatchNorm1d(16)
        self.maxp =  nn.MaxPool1d(10)
        self.layer2 = nn.Conv1d(16, 32, self.kernel_size)
        self.bnorm2 = nn.BatchNorm1d(32)
        self.maxp2 =  nn.MaxPool1d(7)
        self.layer3 = nn.Conv1d(32, 64, self.kernel_size)
        self.bnorm3 = nn.BatchNorm1d(64)
        self.maxp3 =  nn.MaxPool1d(3)
        self.rnn = nn.LSTM(64, self.target_class, self.cell_count, dropout=self.dropout, batch_first=True)
        self.dense = nn.Linear(39, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.bnorm1(x)
        x = self.maxp(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.bnorm2(x)
        x = self.maxp2(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.bnorm3(x)
        x = self.maxp3(x)
        x = torch.transpose(x, 1, 2)
#        x = x.reshape(self.batch_size, 449, 32)
        x, _ = self.rnn(x)
        x = torch.transpose(x, 1, 2)
#        x = x.reshape(self.batch_size, 4, 449)
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
        f_score = f1(torch.argmax(probs, dim=1), torch.argmax(y, dim=1), average='weighted', num_classes=4)
        return {"loss": loss, "train_accuracy": acc, "f_score" : f_score}
    
    def validation_step(self, batch_data, batch_index):
        x, y = batch_data
        logits = self(x)
        criterion = nn.CrossEntropyLoss()
        probs = torch.softmax(logits, dim=1)
        
        # validation metrics
        acc = accuracy(torch.argmax(probs, dim=1), torch.argmax(y, dim=1))
        loss = criterion(logits, torch.argmax(y, dim=1))
        f_score = f1(torch.argmax(probs, dim=1), torch.argmax(y, dim=1), average='weighted', num_classes=4)
        return {"val_loss": loss, "val_accuracy": acc, "f_score" : f_score}
    
    def test_step(self, batch_data, batch_index):
        x, y = batch_data
        logits = self(x)
        criterion = nn.CrossEntropyLoss()
        probs = torch.softmax(logits, dim=1)
        
        # validation metrics
        acc = accuracy(torch.argmax(probs, dim=1), torch.argmax(y, dim=1))
        loss = criterion(logits, torch.argmax(y, dim=1))
        f_score = f1(torch.argmax(probs, dim=1), torch.argmax(y, dim=1), average='weighted', num_classes=4)
        self.log('test/f1', f_score, prog_bar=True)
        self.log('test/loss', loss, prog_bar=True)
        self.log('test/accuracy', acc, prog_bar=True)
        return {"test_loss": loss, "test_accuracy": acc, "f_score" : f_score}
    
    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack(
            [x["loss"] for x in training_step_outputs]).mean()
        avg_acc = torch.stack(
            [x["train_accuracy"] for x in training_step_outputs]).mean()
        avg_f = torch.stack(
            [x["f_score"] for x in training_step_outputs]).mean()
        self.log("ptl/train_f_score", avg_f)
        self.log("ptl/train_loss", avg_loss)
        self.log("ptl/train_accuracy", avg_acc)
    
    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack(
            [x["val_loss"] for x in validation_step_outputs]).mean()
        avg_acc = torch.stack(
            [x["val_accuracy"] for x in validation_step_outputs]).mean()
        avg_f = torch.stack(
            [x["f_score"] for x in validation_step_outputs]).mean()
        self.log("ptl/f_score", avg_f)
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def test_epoch_end(self, test_step_outputs):
        avg_loss = torch.stack(
            [x["test_loss"] for x in test_step_outputs]).mean()
        avg_acc = torch.stack(
            [x["test_accuracy"] for x in test_step_outputs]).mean()
        avg_f = torch.stack(
            [x["f_score"] for x in test_step_outputs]).mean()
        self.log("ptl/test_f_score", avg_f)
        self.log("ptl/test_loss", avg_loss)
        self.log("ptl/test_accuracy", avg_acc)


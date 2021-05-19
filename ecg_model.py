from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.nn import functional as F
import torch

class MLECG(LightningModule):

    def __init__(self):
        super().__init__()
    
        self.layer1 = nn.Conv1d(1, 10, 10)
        self.bnorm1 = nn.BatchNorm1d(1, 0.001, 0.99, affine=False)
        self.maxp1 =  nn.MaxPool1d(18)
    
    def forward(self, x):
        batch_size, length = x.size()

        x = self.layer_1(x)
        x = F.relu(x)

        return x
    
m = nn.Conv1d(1, 1, 10)
input = torch.randn(64, 1, 9000)
output = m(input)
m2 = nn.BatchNorm1d(1, 0.001, 0.99, affine=False)
m3 = nn.MaxPool1d(18)
rnn = nn.LSTM(1, 100, 1, dropout=0.1)
m4 = nn.BatchNorm1d(1, 0.001, 0.99, affine=False)
rnn2 = nn.LSTM(1, 100, 1, dropout=0.1)
m5 = nn.BatchNorm1d(1, 0.001, 0.99, affine=False)
rnn3 = nn.LSTM(1, 100, 1, dropout=0.1)
m6 = nn.BatchNorm1d(1, 0.001, 0.99, affine=False)
out = nn.Linear(128, 10)
output = m2(output)
output = m3(output)
output = rnn(output)
output = m4(output)
output = rnn2(output)
output = m5(output)
output = rnn3(output)
output = m6(output)


print(output.size())
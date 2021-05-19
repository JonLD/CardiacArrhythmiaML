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

#input = torch.randn(64, 1, 9000)
#m = nn.Conv1d(1, 1, 10)
#m2 = nn.BatchNorm1d(1, 0.001, 0.99, affine=False)
#m3 = nn.MaxPool1d(18)
#rnn = nn.LSTM(1, 5, 3, dropout=0.1, batch_first=True)
#out = nn.Linear(5, 1)
#output = m(input)
#output = m2(output)
#output = m3(output)
#output = output.reshape(64,499,1)
#output, (hn, cn) = rnn(output)
#output = out(output)

input = torch.randn(64, 1, 9000)
m = nn.Conv1d(1, 1, 11)
m2 = nn.BatchNorm1d(1, 0.001, 0.99, affine=False)
m3 = nn.MaxPool1d(10)
rnn = nn.LSTM(1, 5, 3, dropout=0.1, batch_first=True)
output = m(input)
output = m2(output)
output = m3(output)

print(output.size())

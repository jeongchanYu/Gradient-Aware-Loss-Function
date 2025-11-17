import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Module, ModuleList, Sequential, LSTM, Linear, LeakyReLU, Dropout, Sigmoid


class SE(Module):
    def __init__(self,):
        super().__init__()
        self.lstm = LSTM(input_size=257, hidden_size=200, num_layers=2, bidirectional=True,batch_first=True)
        self.fc = nn.Sequential(
            Linear(400,300),
            LeakyReLU(),
            Dropout(0.05),
            Linear(300, 257),
            Sigmoid()
        )


    def forward(self, noisy):
        m = torch.mean(noisy, dim=1, keepdim=True)
        s = torch.std(noisy, dim=1, keepdim=True) + 1e-12
        norm = (noisy - m) / s
        mask = self.lstm(norm.transpose(1, 2))[0]
        mask = self.fc(mask).transpose(1, 2)
        mask = torch.clamp(mask, min=0.05)
        return mask * noisy
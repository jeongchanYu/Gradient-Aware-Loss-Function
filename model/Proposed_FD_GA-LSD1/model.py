import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Module, ModuleList, Sequential, Conv1d, LSTM, ReLU


class LSTM1d(Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        return self.lstm(x.transpose(1, 2))[0].transpose(1, 2)


class SE(Module):
    def __init__(self, fft_size, hidden_size):
        super().__init__()

        self.lstm = Sequential(LSTM1d(fft_size//2, hidden_size, 2),
                               Conv1d(hidden_size*2, fft_size//2, 1),
                               ReLU())

    def forward(self, noisy):
        return self.lstm(noisy)

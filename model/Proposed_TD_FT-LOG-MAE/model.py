import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Module, ModuleList, Sequential, Conv1d, ConvTranspose1d, LSTM, Sigmoid


class LSTM1d(Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        return self.lstm(x.transpose(1, 2))[0].transpose(1, 2)


class SE(Module):
    def __init__(self, kernel_size, channel_size, hidden_size):
        super().__init__()

        self.encoder = Conv1d(1, channel_size, kernel_size, kernel_size//2, bias=False)
        self.lstm = Sequential(LSTM1d(channel_size, hidden_size, 2),
                               Conv1d(hidden_size*2, channel_size, 1),
                               Sigmoid())
        self.decoder = ConvTranspose1d(channel_size, 1, kernel_size, kernel_size//2, bias=False)

    def forward(self, noisy):
        latent = self.encoder(noisy)
        mask = self.lstm(latent)
        clean = self.decoder(latent*mask)
        return clean
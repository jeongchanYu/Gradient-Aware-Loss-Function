import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import torch
from torchinfo import summary
from config import *
import model


device = torch.device('cpu')

se_model = model.SE(fft_size, hidden_size).to(device)
se_model.eval()

summary(se_model)
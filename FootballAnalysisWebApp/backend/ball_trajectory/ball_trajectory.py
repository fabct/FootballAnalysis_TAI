import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class BallTrajectoryRNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=200, output_dim=4, num_layers=2):
        super(BallTrajectoryRNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out



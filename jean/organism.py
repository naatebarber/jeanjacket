import torch
import torch.nn as nn


class Organism(nn.Module):
    def __init__(self, d_in, d_ff, d_out, device=torch.device("cpu")):
        super(Organism, self).__init__()
        self.device = device
        self.d_in = d_in
        self.d_ff = d_ff
        self.fc1 = nn.Linear(d_in, d_ff).to(device)
        self.activ = nn.ReLU().to(device)
        self.fc2 = nn.Linear(d_ff, d_out).to(device)

    def forward(self, x):
        x = self.activ(self.fc1(x))
        x = self.fc2(x)
        return x

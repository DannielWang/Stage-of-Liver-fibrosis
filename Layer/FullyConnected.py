import torch
import torch.nn as nn
import torch.nn.functional as F


class Fullyconnected(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim = 4, bias=False):
        super(Fullyconnected, self).__init__()
        self.fc1 = nn.Linear(in_dim, n_hidden_1, bias)
        self.fc1 = nn.Linear(n_hidden_1, n_hidden_2, bias)
        self.fc3 = nn.Linear(n_hidden_2, out_dim, bias)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out, inplace=False)
        out = self.fc2(out)
        out = F.relu(out, inplace=False)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)
        return out

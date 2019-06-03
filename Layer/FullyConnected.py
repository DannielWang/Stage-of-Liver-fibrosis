import torch
import torch.nn as nn
import torch.nn.functional as F


class Fullyconnected(nn.Module):
    def __init__(self, w_nodes, h_nodes, node_hidden_layer):
        super(Fullyconnected, self).__init__()
        self.fc1 = nn.Linear(w_nodes * h_nodes, node_hidden_layer)
        self.fc2 = nn.Linear(node_hidden_layer, node_hidden_layer)
        self.fc3 = nn.Linear(node_hidden_layer, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out, inplace=False)
        out = self.fc2(out)
        out = F.relu(out, inplace=False)
        out = self.fc3(out)
        out = F.log_softmax(out)
        return out

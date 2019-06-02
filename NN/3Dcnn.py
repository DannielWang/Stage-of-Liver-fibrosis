import torch
import torch.nn as nn
import torch.nn.functional as F
from Layer import ContractC
from Layer import FullyConnected


class CNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(CNN, self).__init__()

        self.contract1 = ContractC.Contract(in_channel, out_channel, kernel_size)
        self.contract2 = ContractC.Contract(out_channel, out_channel >> 1,  kernel_size)
        self.contract3 = ContractC.Contract(out_channel >> 1, (out_channel >> 1) >> 1, kernel_size)
        self.fcl = FullyConnected.Fullyconnected()

    def forward(self, x):
        out = self.contract1(x)
        out = F.relu(out)
        out = self.contract2(out)
        out = F.relu(out)
        out = self.contract3(out)
        out = F.relu(out)
        out = self.fcl(out)
        return out

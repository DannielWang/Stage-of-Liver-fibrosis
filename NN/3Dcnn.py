import torch
import torch.nn as nn
import torch.nn.functional as F
from Layer import ContractC
from Layer import FullyConnected


class CNN(nn.Module):
    def __init__(self, kernel_size=7):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv3d(7, 33, kernel_size)
        self.conv2 = nn.Conv3d(33, 46, kernel_size)
        self.conv3 = nn.Conv3d(46, 78, kernel_size)
        self.conv4 = nn.Conv3d(78, 128, kernel_size)
        self.max_pool1 = nn.MaxPool3d(2)
        self.max_pool2 = nn.MaxPool3d(3)
        # self.contract1 = ContractC.Contract(in_channel, out_channel, kernel_size)
        # self.contract2 = ContractC.Contract(out_channel, out_channel >> 1, kernel_size)
        # self.contract3 = ContractC.Contract(out_channel >> 1, (out_channel >> 1) >> 1, kernel_size)
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

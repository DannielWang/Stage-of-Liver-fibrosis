import torch
import torch.nn as nn
import torch.nn.functional as F


class Contract(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(Contract, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, in_channel, kernel_size)
        self.conv2 = nn.Conv3d(in_channel, out_channel, kernel_size)
        self.normalization2 = nn.BatchNorm3d(out_channel)
        self.normalization1 = nn.BatchNorm3d(in_channel)
        self.max_pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.normalization1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.normalization2(out)
        return out

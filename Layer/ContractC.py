import torch
import torch.nn as nn
import torch.nn.functional as F


class Contract(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(Contract, self).__init__()

        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size, padding=1)
        self.conv2 = nn.Conv3d(out_channel, out_channel >> 1, kernel_size, padding=1)
        self.normalization = nn.BatchNorm2d(out_channel >> 1)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = F.relu(out, inplace=True)
        out = self.normalization(out)
        out = nn.MaxPool2d(out)
        return out

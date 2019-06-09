import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda


class Contract(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Contract, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size)
        self.normalization = nn.BatchNorm2d(out_channel)
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out, inplace=False)
        out = self.normalization(out)
        out = self.conv2(out)
        out = F.relu(out, inplace=False)
        out = self.normalization(out)
        return out

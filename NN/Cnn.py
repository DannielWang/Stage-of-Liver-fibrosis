import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Layer import ContractC
from Layer import FullyConnected
import json
import torch.cuda as cuda


class CNN(nn.Module):
    def __init__(self, kernel_size=7):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv3d(2, 8, kernel_size)
        self.conv2 = nn.Conv3d(8, 16, kernel_size)
        self.conv3 = nn.Conv3d(16, 32, kernel_size)
        self.conv4 = nn.Conv3d(32, 64, kernel_size)
        self.conv5 = nn.Conv3d(64, 128, kernel_size)
        self.max_pool1 = nn.MaxPool3d(2)
        self.max_pool2 = nn.MaxPool3d(3)
        # self.contract1 = ContractC.Contract(in_channel, out_channel, kernel_size)
        # self.contract2 = ContractC.Contract(out_channel, out_channel >> 1, kernel_size)
        # self.contract3 = ContractC.Contract(out_channel >> 1, (out_channel >> 1) >> 1, kernel_size)
        self.fcl = FullyConnected.Fullyconnected()

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = F.relu(out)
        out = self.fcl(out)
        return out


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    net = CNN()
    cirterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.004)
    datadir = ''
    labeldir = ''
    datalist = load_json('phase_liverfibrosis.json')
    dataset = {}

if cuda.is_available():
    net.cuda()
    cirterion.cuda()
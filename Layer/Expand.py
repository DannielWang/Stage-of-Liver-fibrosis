import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Expand(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Expand, self).__init__()

        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')  # scale_factor means division in number of times
        # self.conv = nn.Conv3d(in_channel, out_channel, kernel_size, padding=1)
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size, )
        self.normalization = nn.BatchNorm3d(out_channel)

    def forward(self, x, left_tensor):
        out = self.upsamp(x)
        # out = self.conv(out)
        # cropping
        #
        if out.shape[2] != left_tensor.shape[2]:
            dz_s = math.floor((left_tensor.shape[2] - out.shape[2]) / 2)
            dz_l = math.ceil((left_tensor.shape[2] - out.shape[2]) / 2)
            dx_s = math.floor((left_tensor.shape[3] - out.shape[3]) / 2)
            dx_l = math.ceil((left_tensor.shape[3] - out.shape[3]) / 2)
            dy_s = math.floor((left_tensor.shape[4] - out.shape[4]) / 2)
            dy_l = math.ceil((left_tensor.shape[4] - out.shape[4]) / 2)
            left_tensor = left_tensor[:, :, dz_s:-dz_l, dx_s:-dx_l, dy_s:-dy_l]
        elif out.shape[3] != left_tensor.shape[3]:
            dz_s = math.floor((left_tensor.shape[2] - out.shape[2]) / 2)
            dz_l = math.ceil((left_tensor.shape[2] - out.shape[2]) / 2)
            dx_s = math.floor((left_tensor.shape[3] - out.shape[3]) / 2)
            dx_l = math.ceil((left_tensor.shape[3] - out.shape[3]) / 2)
            dy_s = math.floor((left_tensor.shape[4] - out.shape[4]) / 2)
            dy_l = math.ceil((left_tensor.shape[4] - out.shape[4]) / 2)
            left_tensor = left_tensor[:, :, dz_s:-dz_l, dx_s:-dx_l, dy_s:-dy_l]
        elif out.shape[4] != left_tensor.shape[4]:
            dz_s = math.floor((left_tensor.shape[2] - out.shape[2]) / 2)
            dz_l = math.ceil((left_tensor.shape[2] - out.shape[2]) / 2)
            dx_s = math.floor((left_tensor.shape[3] - out.shape[3]) / 2)
            dx_l = math.ceil((left_tensor.shape[3] - out.shape[3]) / 2)
            dy_s = math.floor((left_tensor.shape[4] - out.shape[4]) / 2)
            dy_l = math.ceil((left_tensor.shape[4] - out.shape[4]) / 2)
            left_tensor = left_tensor[:, :, dz_s:-dz_l, dx_s:-dx_l, dy_s:-dy_l]
        out = torch.cat((out, left_tensor), dim=1)
        out = self.conv1(out)
        out = F.relu(out, inplace=False)
        out = self.conv2(out)
        out = self.normalization(out)
        out = F.relu(out, inplace=False)
        return out

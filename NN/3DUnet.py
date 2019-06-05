import torch
import torch.nn as nn
import torch.nn.functional as F
from Layer import ContractU
from Layer import Expand
import os
import numpy as np
import torch.optim as optim
import mhd
import json
import tqdm
from skimage import measure


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        kernel_size = 3
        # define the module object of each layer
        self.conv = nn.Conv3d(1, 3, kernel_size)
        self.conv1 = nn.Conv3d(3, 32, kernel_size)
        self.conv2 = nn.Conv3d(32, 64, kernel_size)
        self.contract1 = ContractU.Contract(64, 128, kernel_size)
        self.contract2 = ContractU.Contract(128, 256, kernel_size)
        self.contract3 = ContractU.Contract(256, 512, kernel_size)
        self.expand1 = Expand.Expand(768, 256, kernel_size)
        self.expand2 = Expand.Expand(384, 128, kernel_size)
        self.expand3 = Expand.Expand(192, 64, kernel_size)
        self.max_pool = nn.MaxPool3d(2, stride=2)
        self.dropout = nn.Dropout3d(0.25, inplace=False)
        self.conv3 = nn.Conv3d(64, 3, kernel_size=1)
        self.normaliztion1 = nn.BatchNorm3d(32)
        self.normaliztion2 = nn.BatchNorm3d(64)

    def forward(self, x):
        # Convolution operation
        out = self.conv(x)
        out = self.conv1(out)
        out = self.normaliztion1(out)
        out = F.relu(out, inplace=False)
        out = self.conv2(out)
        out = self.normaliztion2(out)
        out = F.relu(out, inplace=False)
        # Store the tensor of each convoluted layer
        left_tensor1 = out
        out = self.max_pool(out)
        out = self.contract1(out)
        left_tensor2 = out
        out = self.max_pool(out)
        out = self.contract2(out)
        left_tensor3 = out
        out = self.max_pool(out)
        out = self.contract3(out)
        # Upsampling operation
        out = self.expand1(out, left_tensor3)
        out = self.expand2(out, left_tensor2)
        out = self.dropout(out)
        out = self.expand3(out, left_tensor1)
        out = self.conv3(out)

        return out



def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def split_dataset(datalist, k):
    groups = []
    for key in ['Liver fibrosis in stage 1', 'Liver fibrosis in stage 2', 'Liver fibrosis in stage 3',
                'Liver fibrosis in stage 4']:
        IDs = datalist[key]
        groups.append(np.array_split(np.random.permutation(np.array(IDs, dtype=str)), k))
    all_IDs = [e for v in datalist.values() for e in v]
    ikeda_IDs = set([ID[:-1] for ID in all_IDs if ID.startswith('ikeda_image')])
    ikeda_groups = np.array_split(np.random.permutation(np.array(list(ikeda_IDs), dtype=str)), k)
    for suffix in ['E', 'L']:
        groups.append([[ID + suffix for ID in g if ID + suffix in all_IDs] for g in ikeda_groups])
    return [sorted(np.concatenate([g[i] for g in groups]).tolist()) for i in range(k)]



if __name__ == '__main__':
    ########################################################
    net = Unet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01,weight_decay=0.001 )
    datadir = '../dataset/Cases/Images'
    labeldir = '../dataset/Cases/Labels'
    datalist = load_json('phase_liverfibrosis.json')
    groups = split_dataset(datalist, 4)
    dataset = {}
    y_shape = [512, 512]
    Image_shape = [512, 512]
    # n_classes = 1

    for ID in tqdm.tqdm(os.listdir(datadir), desc='Loading images'):
        if os.path.isfile(datadir + '\\' + ID):
            out = ID.split('.')
            if len(ID) >= 2:
                if out[1] == 'mhd':
                    image = mhd.read(os.path.join(datadir, ID))[0]
                    vmask = mhd.read(os.path.join(labeldir, ID[:-9] + 'label.mhd'))[0]
                    data = {}
                    data['x'] = np.expand_dims((image / 255.0).astype(np.float32), 0)
                    data['y'] = np.expand_dims(vmask, 0)
                    dataset = data['x']
                    dataset = torch.Tensor(dataset)
                    dataset = dataset.unsqueeze(0)
    for epoch in range(4):
        running_loss = 0.0
        for i,x in enumerate(dataset, 0):
            inputs, labels = x
            inputs = inputs.unsqueeze(0)
            labels = labels.unsqueeze(0)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs,labels)
            loss.backward(retain_graph=True)
            print("epoch = {}, loss = {:.5f}".format(ID + 1, loss.data))
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d,%5d] loss: %.3f'%(epoch+1,i+1,running_loss/2000))
                running_loss = 0.0
    print('Finished Training')

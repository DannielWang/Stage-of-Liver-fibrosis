# matplotlib inline
# config InlineBackend.figure_format = 'retina'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from Layer import ContractU2d
from Layer import ExpandU2d
from Layer import FullyConnected
import os
import numpy as np
import json
import pickle
from skimage import measure
from datetime import datetime
from Function import Dataset
from scipy import ndimage
import sys
import shutil
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        kernel_size = 3

        # Unet
        self.contract1 = ContractU2d.Contract(1, 64, kernel_size)
        self.contract2 = ContractU2d.Contract(64, 128, kernel_size)
        self.contract3 = ContractU2d.Contract(128, 256, kernel_size)
        self.contract4 = ContractU2d.Contract(256, 512, kernel_size)
        self.contract5 = ContractU2d.Contract(512, 1024, kernel_size)
        self.expand1 = ExpandU2d.Expand(1024, 512, kernel_size)
        self.expand2 = ExpandU2d.Expand(512, 256, kernel_size)
        self.expand3 = ExpandU2d.Expand(256, 128, kernel_size)
        self.expand4 = ExpandU2d.Expand(128, 64, kernel_size)
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.25, inplace=False)
        self.conv = nn.Conv2d(64, 2, kernel_size=1)

        # 3D CNN
        self.conv1 = nn.Conv3d(2, 8, kernel_size)
        self.conv2 = nn.Conv3d(8, 16, kernel_size)
        self.conv3 = nn.Conv3d(16, 32, kernel_size)
        self.conv4 = nn.Conv3d(32, 64, kernel_size)
        self.conv5 = nn.Conv3d(64, 128, kernel_size)
        self.max_pool1 = nn.MaxPool3d(2)
        self.max_pool2 = nn.MaxPool3d(3)
        self.fcl = FullyConnected.Fullyconnected(9248,1000,300)

    def forward(self, x):
        # Unet
        # Convolution operation
        out = self.contract1(x)
        # Store the tensor of each convoluted layer
        left_tensor1 = out
        out = self.max_pool(out)
        out = self.contract2(out)
        left_tensor2 = out
        out = self.max_pool(out)
        out = self.contract3(out)
        left_tensor3 = out
        out = self.max_pool(out)
        out = self.contract4(out)
        left_tensor4 = out
        out = self.max_pool(out)
        out = self.contract5(out)
        # Upsampling operation
        out = self.expand1(out, left_tensor4)
        out = self.expand2(out, left_tensor3)
        out = self.expand3(out, left_tensor2)
        out = self.dropout(out)
        out = self.expand4(out, left_tensor1)
        out = self.conv(out)

        # 3D CNN
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = F.relu(out)
        out = self.fcl(out)

        return out


def downsample(img, size):
    down_img = ndimage.interpolation.zoom(img, (1, size / img.shape[1], size / img.shape[2]))

    return down_img


def largest_CC(image, n=1):
    labels = measure.label(image, connectivity=3, background=0)
    area = np.bincount(labels.flat)
    if (len(area) > 1):
        return labels == (np.argmax(area[1:]) + 1)
    else:
        return np.zeros(labels.shape, np.bool)


def refine_labels(labels):
    refined = np.zeros_like(labels)
    for i in range(1, np.max(labels) + 1):
        cc = largest_CC(labels == i)
        refined[cc] = i
    return refined


def save_object(obj, filename):
    if os.path.splitext(filename)[1] == '.json':
        with open(filename, 'w') as f:
            json.dump(obj, f, indent=2)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def split_dataset(datalist, k):
    groups = []
    for key in ['Liver fibrosis in stage 2', 'Liver fibrosis in stage 3',
                'Liver fibrosis in stage 4']:
        IDs = datalist[key]
        groups.append(np.array_split(np.random.permutation(np.array(IDs, dtype=str)), k))
    all_IDs = [e for v in datalist.values() for e in v]
    ikeda_IDs = set([ID[:-1] for ID in all_IDs if ID.startswith('ikeda_image')])
    ikeda_groups = np.array_split(np.random.permutation(np.array(list(ikeda_IDs), dtype=str)), k)
    for suffix in ['E', 'L']:
        groups.append([[ID + suffix for ID in g if ID + suffix in all_IDs] for g in ikeda_groups])
    return [sorted(np.concatenate([g[i] for g in groups]).tolist()) for i in range(k)]


#################################################################################################################
if __name__ == '__main__':
    ########################################################
    net = Unet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00015, betas=(0.9, 0.999), weight_decay=0.004)
    datadir = '../dataset/Cases/Images'
    labeldir = '../dataset/Cases/Labels'
    datalist = load_json('phase_liverfibrosis.json')
    dataset = {}

    trainloader, testloader = Dataset.ShipDataset(datadir, labeldir, augment=None)
    ship_train_loader = DataLoader(trainloader, batch_size=1, num_workers=4, shuffle=True)
    imgsize = 256
    labelsize = 68

    if cuda.is_available():
        net.cuda()
        criterion.cuda()

    result_basedir = 'train_' + datetime.today().strftime("%y%m%d_%H%M%S")
    os.makedirs(result_basedir, exist_ok=True)
    shutil.copyfile(sys.argv[0], os.path.join(result_basedir, os.path.basename(sys.argv[0])))  # copy this script

    all_IDs = sorted(dataset.keys())

    #
    epochs = 1
    steps = 0
    print_every = 10
    train_losses, test_losses = [], []
    k_fold = 2
    for epoch in range(epochs):
        groups = split_dataset(datalist, k_fold)
        exp_dir = os.path.join(result_basedir, 'exp{0}'.format(epoch))
        os.makedirs(exp_dir, exist_ok=True)
        result_img_dir = os.path.join(exp_dir, 'img')
        os.makedirs(result_img_dir, exist_ok=True)
        save_object([g for g in groups], os.path.join(exp_dir, 'groups.json'))
        running_loss = 0.0
        loss = []
        ERROR_Train = []
        JIs, DCs = {}, {}
        refined_JIs, refined_DCs = {}, {}
        net.train()
        for i, (image, labels) in enumerate(trainloader):
            steps += 1
            optimizer.zero_grad()
            real_cpu = torch.Tensor(downsample(image, imgsize))
            label_cpu = torch.Tensor(downsample(labels, labelsize))
            if cuda.is_available():
                real_cpu = real_cpu.cuda()
                label_cpu = label_cpu.cuda()

                inputs = real_cpu
                labels = label_cpu

                inputv = Variable(inputs)
                labelv = Variable(labels)
                inputv = inputv.unsqueeze(1)
                labelv = labelv.unsqueeze(1)
                output = net(inputv)

                loss = criterion(output, labelv)
                loss.backward(retain_graph=True)
                print("epoch = {}, loss = {:.5f}".format(epoch + 1, loss.data))
                optimizer.step()
                out = output.data.cpu()
                running_loss += loss.item()
                if steps % print_every == 0:
                    test_losses = 0
                    accuracy = 0
                    net.eval()
                    with torch.no_grad():
                        for image, labels in testloader:
                            image, labels = image.cuda(), labels.cuda()
                            logps = Unet(image)
                            batch_loss = criterion(logps, labels)
                            test_losses += batch_loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            train_losses.append(running_loss / len(trainloader))
                            test_losses.append(test_losses / len(testloader))
                            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
                            running_loss = 0.0
                            net.train()
    torch.save(net, 'train.pth')
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()
    result_dir = os.path.join(exp_dir, 'g{0}'.format(epoch))
    os.makedirs(result_dir, exist_ok=True)

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from Layer import ContractU2d
from Layer import ExpandU2d
import os
import numpy as np
import mhd
import json
import pickle
import tqdm
from skimage import measure
from datetime import datetime
from Function import Dataset
from scipy import ndimage
from scipy import misc
import sys
import shutil



class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        kernel_size = 3
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

    def forward(self, x):
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

        return F.softmax(out, dim=1)


def downsample(img, size):
    down_img = ndimage.interpolation.zoom(img, (1, size / img.shape[1], size / img.shape[2]))

    return down_img


def JaccardIndex(a, b):
    return np.sum(a & b) / np.sum(a | b)


def DiceCoeff(a, b):
    return 2 * np.sum(a & b) / (np.sum(a) + np.sum(b))


def evaluate(label1, label2):
    max_label = max(np.max(label1), np.max(label2))
    JIs = []
    DCs = []
    for i in range(1, max_label + 1):
        a = label1 == i
        b = label2 == i
        JIs.append(JaccardIndex(a, b))
        DCs.append(DiceCoeff(a, b))
    return JIs, DCs

def predict(self, x, axis=0, params=None):

    if params is None:
        params = self.params
    assert params is not None
    if len(params) != 2:
        raise ValueError('Parameters are defined by 2 sets.')

    origin, direction = params

    if direction[axis] == 0:
        # line parallel to axis
        raise ValueError('Line parallel to axis %s' % axis)

    l = (x - origin[axis]) / direction[axis]
    data = origin + l[..., np.newaxis] * direction
    return data

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
    # scheduler = optim.lr_shcduker.ExponentialLR(optimizer, gamma=0.997)
    datadir = '../dataset/Cases/Images'
    labeldir = '../dataset/Cases/Labels'
    datalist = load_json('phase_liverfibrosis.json')
    # groups = split_dataset(datalist, 2)
    dataset = {}
    # n_classes = 4
    for ID in tqdm.tqdm(os.listdir(datadir), desc='Loading images'):
        if os.path.isfile(datadir + '\\' + ID):
            out = ID.split('.')
            if len(ID) >= 2:
                if out[1] == 'mhd':
                    original = mhd.read(os.path.join(datadir, ID))[0]
                    label = mhd.read(os.path.join(labeldir, ID[:-9] + 'label.mhd'))[0]
                    data = {}
                    data['x'] = np.expand_dims((original / 255.0).astype(np.float32), -1)
                    data['y'] = np.expand_dims(label, -1)
                    dataset[ID] = data
        # n_classes = max(n_classes, np.max(label) + 1)

    ship_train_dataset = Dataset.ShipDataset(datadir, labeldir, augment=None)
    ship_train_loader = DataLoader(ship_train_dataset, batch_size=1, num_workers=4, shuffle=True)
    imgsize = 256
    labelsize = 68

    if cuda.is_available():
        net.cuda()
        criterion.cuda()

    result_basedir = 'unet_train_' + datetime.today().strftime("%y%m%d_%H%M%S")
    os.makedirs(result_basedir, exist_ok=True)
    shutil.copyfile(sys.argv[0], os.path.join(result_basedir, os.path.basename(sys.argv[0])))  # copy this script

    all_IDs = sorted(dataset.keys())

    #
    k_fold = 2
    for epochs in range(4):
        groups = split_dataset(datalist, k_fold)
        exp_dir = os.path.join(result_basedir, 'exp{0}'.format(epochs))
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
        for i, (image, labels) in enumerate(ship_train_loader):
            result_dir = os.path.join(exp_dir, 'g{0}'.format(i))
            os.makedirs(result_dir, exist_ok=True)

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
                print("epoch = {}, loss = {:.5f}".format(epochs + 1, loss.data))
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d,%5d] loss: %.3f' % (epochs + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

                # save model
                # with open(os.path.join(result_dir, 'unet_model.json'), 'w') as f:
                #     f.write(net.to_json())
                net.state_dict()
                torch.save(net, os.path.join(result_dir, 'unet_model_weights.h5'))

                for i, test_IDs in enumerate(groups):
                    for i in range(len(test_IDs)):
                        # filename = result_img_dir + 'ikeda_%d.mhd' % i
                        refined = Variable(torch.Tensor(output.cpu()), requires_grad = True)
                        refined = refined.detach().numpy()
                        # misc.toimage(refined, cmin=0.0, cmax=...).save(filename)  # save img
                        img_mask = np.array(refined) * 255  ##img_mask is Binary Image
                        img_mask = cv2.merge(img_mask)
                        cv2.imwrite(result_img_dir + 'ikeda_%d.mhd' % i , img_mask)

                #     train_ID = [ID for ID in dataset.keys()]
                # for test_ID in tqdm.tqdm(train_ID, desc='Testing'):
                #     x_test = np.fromstring(dataset[test_ID]['x'])
                #     predict_y = predict(self=None,x=x_test)
                #     predict_label = np.argmax(predict_y, axis=3).astype(np.uint8)
                #     mhd.write(os.path.join(result_img_dir, test_IDs + '.mhd'), predict_label)

                    # JIs[test_IDs], DCs[test_IDs] = evaluate(predict_label, np.squeeze(dataset[test_IDs]['y']))
                    # refined = refine_labels(predict_label)
                    # mhd.write(os.path.join(result_img_dir, 'refined_' + test_IDs + '.mhd'), refined)
                    # refined_JIs[test_IDs], refined_DCs[test_IDs] = evaluate(refined, np.squeeze(dataset[test_IDs]['y']))

            # np.savetxt(os.path.join(exp_dir, 'refined_JI.csv'), np.stack([refined_JIs[ID] for ID in all_IDs]),
            #            delimiter=",", fmt='%g')
            # np.savetxt(os.path.join(exp_dir, 'refined_Dice.csv'), np.stack([refined_DCs[ID] for ID in all_IDs]),
            #            delimiter=",", fmt='%g')

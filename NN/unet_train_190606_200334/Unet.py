import torch
import torch.nn as nn
import torch.nn.functional as F
from Layer import ContractU2d
from Layer import ExpandU2d
import os
import numpy as np
import torch.optim as optim
import mhd
import json
import pickle
import tqdm
from skimage import measure
from datetime import datetime
import sys, shutil


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        kernel_size = 3
        # define the module object of each layer
        # self.conv = nn.Conv3d(1, 3, kernel_size)
        # self.conv1 = nn.Conv3d(3, 32, kernel_size)
        # self.conv2 = nn.Conv3d(32, 64, kernel_size)
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
        # Upsampling operation
        out = self.expand1(out, left_tensor4)
        out = self.expand2(out, left_tensor3)
        out = self.expand3(out, left_tensor2)
        out = self.dropout(out)
        out = self.expand4(out, left_tensor1)
        out = self.conv(out)

        return F.softmax(out)


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


#################################################################################################################
if __name__ == '__main__':
    ########################################################
    net = Unet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)
    datadir = '../dataset/Cases/Images'
    labeldir = '../dataset/Cases/Labels'
    datalist = load_json('phase_liverfibrosis.json')
    groups = split_dataset(datalist, 2)
    dataset = {}
    y_shape = [512, 512]
    Image_shape = [512, 512]
    n_classes = 4

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
                    dataset[ID] = data
                    n_classes = max(n_classes, np.max(vmask) + 1)

    result_basedir = 'unet_train_' + datetime.today().strftime("%y%m%d_%H%M%S")
    os.makedirs(result_basedir, exist_ok=True)

    shutil.copyfile(sys.argv[0], os.path.join(result_basedir, os.path.basename(sys.argv[0])))  # copy this script

    all_IDs = sorted(dataset.keys())
    # datalist = {}
    # temp = []
    # for key in datalist.keys():
    #     for ID in datalist[key]:
    #         if ID in all_IDs:
    #             temp.append(ID)
    # dataset['key'] = temp

    k_fold = 2
    for exp_no in range(1):
        groups = split_dataset(dataset, k_fold)
        exp_dir = os.path.join(result_basedir, 'exp{0}'.format(exp_no))
        os.makedirs(exp_dir, exist_ok=True)
        result_img_dir = os.path.join(exp_dir, 'img')
        os.makedirs(result_img_dir, exist_ok=True)
        save_object([g for g in groups], os.path.join(exp_dir, 'groups.json'))
        JIs, DCs = {}, {}
        refined_JIs, refined_DCs = {}, {}
        print('pre_groups')
        print(groups)
        for group_no, test_IDs in enumerate(groups):
            result_dir = os.path.join(exp_dir, 'g{0}'.format(group_no))
            os.makedirs(result_dir, exist_ok=True)
            # model = create_model([128, 128, 1], n_classes)
            # opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-3)
            # model.compile(opt, 'sparse_categorical_crossentropy')
            # es = keras.callbacks.EarlyStopping(monitor='loss', patience=1, verbose=1, mode='auto')
            print('groups')
            print(groups)
            print('test_IDs')
            print(test_IDs)
            inputs = test_IDs
            labels = test_IDs

            optimizer.zero_grad()
            inputs = torch.Tensor(inputs)
            labels = torch.Tensor(labels)
            print('inputs')
            print(inputs.shape)
            print('labels')
            print(labels.shape)
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            print("epoch = {}, loss = {:.5f}".format(ID + 1, loss.data))
            optimizer.step()

            running_loss += loss.item()
            if group_no % 2000 == 1999:
                print('[%d,%5d] loss: %.3f' % (exp_no + 1, group_no + 1, running_loss / 2000))
                running_loss = 0.0
            print('Finished Training')
            train_IDs = [ID for ID in dataset.keys() if ID not in test_IDs]
            x_train = np.concatenate([dataset[ID]['x'] for ID in train_IDs])
            y_train = np.concatenate([dataset[ID]['y'] for ID in train_IDs])
            print('test', test_IDs)
            print('train', train_IDs)
            epochs = 4
            outputs.fit(x_train, y_train, batch_size=4, epochs=epochs)

    # save model
    with open(os.path.join(result_dir, 'unet_model.json'), 'w') as f:
        f.write(outputs.to_json())
    outputs.save_weights(os.path.join(result_dir, 'unet_model_weights.h5'))

    for test_ID in tqdm.tqdm(test_IDs, desc='Testing'):
        x_test = dataset[test_ID]['x']
        predict_y = outputs.predict(x_test, batch_size=4, verbose=False)

        predict_label = np.argmax(predict_y, axis=3).astype(np.uint8)
        mhd.write(os.path.join(result_img_dir, test_ID + '.mha'), predict_label)

        JIs[test_ID], DCs[test_ID] = evaluate(predict_label, np.squeeze(dataset[test_ID]['y']))
        refined = refine_labels(predict_label)
        mhd.write(os.path.join(result_img_dir, 'refined_' + test_ID + '.mha'), refined)
        refined_JIs[test_ID], refined_DCs[test_ID] = evaluate(refined, np.squeeze(dataset[test_ID]['y']))

    np.savetxt(os.path.join(exp_dir, 'JI.csv'), np.stack([JIs[ID] for ID in all_IDs]), delimiter=",", fmt='%g')
    np.savetxt(os.path.join(exp_dir, 'Dice.csv'), np.stack([DCs[ID] for ID in all_IDs]), delimiter=",", fmt='%g')
    np.savetxt(os.path.join(exp_dir, 'refined_JI.csv'), np.stack([refined_JIs[ID] for ID in all_IDs]), delimiter=",",
               fmt='%g')
    np.savetxt(os.path.join(exp_dir, 'refined_Dice.csv'), np.stack([refined_DCs[ID] for ID in all_IDs]), delimiter=",",
               fmt='%g')

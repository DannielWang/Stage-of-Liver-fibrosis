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


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        kernel_size = 3
        # define the module object of each layer
        self.conv1 = nn.Conv3d(3, 32, 512, 512, 84, kernel_size)
        self.conv2 = nn.Conv3d(32, 64, 512, 512, 84, kernel_size)
        self.contract1 = ContractU.Contract(64, 128, 512, 512, 84, kernel_size)
        self.contract2 = ContractU.Contract(128, 256, 512, 512, 84, kernel_size)
        self.contract4 = ContractU.Contract(256, 512, 512, 512, 84, kernel_size)
        self.expand1 = Expand.Expand(512, 256, 512, 512, 84, kernel_size)
        self.expand2 = Expand.Expand(256, 128, 512, 512, 84, kernel_size)
        self.expand3 = Expand.Expand(128, 64, 512, 512, 84, kernel_size)
        self.max_pool = nn.MaxPool3d(2, stride=2)
        self.dropout = nn.Dropout3d(0.25, inplace=True)
        self.conv3 = nn.Conv3d(64, 3, 512, 512, 84, kernel_size=1)
        self.normaliztion1 = nn.BatchNorm3d(32)
        self.normaliztion2 = nn.BatchNorm3d(64)

    def forward(self, x):
        # Convolution operation
        out = self.conv1(x)
        out = self.normaliztion1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.normaliztion2(out)
        out = F.relu(out)
        # Store the tensor of each convoluted layer
        left_tensor1 = out
        out = self.max_pool(out)
        out = self.contract1(out)
        left_tensor2 = out
        out = self.max_pool(out)
        out = self.contract2(out)
        left_tensor3 = out
        out = self.max_pool(out)
        # Upsampling operation
        out = self.expand1(out, left_tensor3)
        out = self.expand2(out, left_tensor2)
        out = self.dropout(out)
        out = self.expand3(out, left_tensor1)
        out = self.conv(out)

        return out
        # Dropout


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


# def split_dataset(datalist, k):
#     groups = []
#     for key in ['NonContrast', 'Venous']:
#         IDs = datalist[key]
#         groups.append(np.array_split(np.random.permutation(np.array(IDs, dtype=str)), k))
#     all_IDs = [e for v in datalist.values() for e in v]
#     Osaka_IDs = set([ID[:-1] for ID in all_IDs if ID.startswith('Osaka')])
#     osaka_groups = np.array_split(np.random.permutation(np.array(list(Osaka_IDs), dtype=str)), k)
#     for suffix in ['E', 'L']:
#         groups.append([[ID + suffix for ID in g if ID + suffix in all_IDs] for g in osaka_groups])
#     return [sorted(np.concatenate([g[i] for g in groups]).tolist()) for i in range(k)]


if __name__ == '__main__':
    net = Unet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    datadir = r'C:\Users\wangdian\PycharmProjects\pytorch\Liver dataset\Data'
    datalist = load_json('phase_liverfibrosis.json')
    # groups = split_dataset(datalist, 2)
    # dataset = {}
    # y_shape = [512, 512]
    # original_shape = [512, 512]
    # n_classes = 1
    # for ID in tqdm.tqdm(os.listdir(datadir), desc='Loading images'):
    #     image = mhd.read(os.path.join(datadir, ID, 'image.mhd'))[0]
    #     vmask = mhd.read(os.path.join(datadir, ID, 'vmask.mhd'))[0]
    #     data = {}
    #     er = image / 255.0
    #     print(er)
    #     data['x'] = np.expand_dims((image / 255.0).astype(np.float32), -1)
    #     data['y'] = np.expand_dims(vmask, -1)
    #
    #     for i in range(100):
    #         print(data['y'][i].max())
    #     dataset[ID] = data
    #     n_classes = max(n_classes, np.max(vmask) + 1)

    # from datetime import datetime
    #
    # result_basedir = 'unet_train_' + datetime.today().strftime("%y%m%d_%H%M%S")
    # os.makedirs(result_basedir, exist_ok=True)
    # import sys, shutil
    #
    # shutil.copyfile(sys.argv[0], os.path.join(result_basedir, os.path.basename(sys.argv[0])))  # copy this script
    #
    # all_IDs = sorted(dataset.keys())
    # datalist = {key: [ID for ID in datalist[key] if ID in all_IDs] for key in datalist.keys()}
    #
    # k_fold = 2
    # for exp_no in range(1):
    #     groups = split_dataset(datalist, k_fold)
    #     exp_dir = os.path.join(result_basedir, 'exp{0}'.format(exp_no))
    #     os.makedirs(exp_dir, exist_ok=True)
    #     result_img_dir = os.path.join(exp_dir, 'img')
    #     os.makedirs(result_img_dir, exist_ok=True)
    #     save_object([g for g in groups], os.path.join(exp_dir, 'groups.json'))
    #     JIs, DCs = {}, {}
    #     refined_JIs, refined_DCs = {}, {}
    #     for group_no, test_IDs in enumerate(groups):
    #         result_dir = os.path.join(exp_dir, 'g{0}'.format(group_no))
    #         os.makedirs(result_dir, exist_ok=True)
    #         model = create_model([128, 128, 1], n_classes)
    #         opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-3)
    #         model.compile(opt, 'sparse_categorical_crossentropy')
    #         es = keras.callbacks.EarlyStopping(monitor='loss', patience=1, verbose=1, mode='auto')
    #         train_IDs = [ID for ID in dataset.keys() if ID not in test_IDs]
    #         x_train = np.concatenate([dataset[ID]['x'] for ID in train_IDs])
    #         y_train = np.concatenate([dataset[ID]['y'] for ID in train_IDs])
    #         print('test', test_IDs)
    #         print('train', train_IDs)
    #         epochs = 4
    #         model.fit(x_train, y_train, batch_size=4, epochs=epochs, callbacks=[es])
    #
    #         # save model
    #         with open(os.path.join(result_dir, 'unet_model.json'), 'w') as f:
    #             f.write(model.to_json())
    #         model.save_weights(os.path.join(result_dir, 'unet_model_weights.h5'))
    #
    #         for test_ID in tqdm.tqdm(test_IDs, desc='Testing'):
    #             x_test = dataset[test_ID]['x']
    #             predict_y = model.predict(x_test, batch_size=4, verbose=False)
    #
    #             predict_label = np.argmax(predict_y, axis=3).astype(np.uint8)
    #             mhd.write(os.path.join(result_img_dir, test_ID + '.mha'), predict_label)
    #
    #             JIs[test_ID], DCs[test_ID] = evaluate(predict_label, np.squeeze(dataset[test_ID]['y']))
    #             refined = refine_labels(predict_label)
    #             mhd.write(os.path.join(result_img_dir, 'refined_' + test_ID + '.mha'), refined)
    #             refined_JIs[test_ID], refined_DCs[test_ID] = evaluate(refined, np.squeeze(dataset[test_ID]['y']))
    #
    #     np.savetxt(os.path.join(exp_dir, 'JI.csv'), np.stack([JIs[ID] for ID in all_IDs]), delimiter=",", fmt='%g')
    #     np.savetxt(os.path.join(exp_dir, 'Dice.csv'), np.stack([DCs[ID] for ID in all_IDs]), delimiter=",", fmt='%g')
    #     np.savetxt(os.path.join(exp_dir, 'refined_JI.csv'), np.stack([refined_JIs[ID] for ID in all_IDs]), delimiter=",",
    #                fmt='%g')
    #     np.savetxt(os.path.join(exp_dir, 'refined_Dice.csv'), np.stack([refined_DCs[ID] for ID in all_IDs]), delimiter=",",
    #                fmt='%g')

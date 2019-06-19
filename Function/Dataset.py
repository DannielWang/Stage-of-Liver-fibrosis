from scipy import ndimage
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import SimpleITK as sitk
import tqdm
import matplotlib.pyplot as plt
import mhd
from torchvision import datasets, transforms, models

def downsample(img, size):
    down_img = ndimage.interpolation.zoom(img, (1, size / img.shape[1], size / img.shape[2]))

    return down_img

class ShipDataset(Dataset):
    def __init__(self, ImagePath, LabelPath, valid_size=.2, augment=None):
        # All this list stored direction of Img.
        self.image_paths = np.array([x.path for x in os.scandir(ImagePath)
                                     if x.name.endswith('image.mhd')])
        self.label_paths = np.array([x.path for x in os.scandir(LabelPath)
                                     if x.name.endswith('label.mhd')])
        self.augment = augment  # Need augment the Image?

        self.images = []
        self.labels = []

        for i in tqdm.tqdm(range(len(self.image_paths)), desc='Loading images'):
            image, image_header = mhd.read(self.image_paths[i])
            label, label_header = mhd.read(self.label_paths[i])
            image = downsample(image, 256)
            label = (downsample(label, 256)).astype(np.uint32)
            image = torch.Tensor(image)
            label = torch.Tensor(label)
            for j in range(image.shape[0]):
                self.images.append(image[j])
                self.labels.append(label[j])

        num_train = len(self.images)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)
        from torch.utils.data.sampler import SubsetRandomSampler
        train_idx, test_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        trainloader = torch.utils.data.DataLoader(image,
                                                  sampler=train_sampler, batch_size=64)
        testloader = torch.utils.data.DataLoader(image,
                                                 sampler=test_sampler, batch_size=64)
        return trainloader,testloader

    def __getitem__(self, index):
        # Load the Image and return it.
        image = self.images[index]
        label = self.labels[index]
        if self.augment:
            image = self.augment(image)
        return torch.Tensor(image), torch.Tensor(label)

    def __len__(self):
        # The number of Image returned
        return len(self.images)

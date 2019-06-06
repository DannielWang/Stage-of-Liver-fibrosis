from torch.utils.data import Dataset
import numpy as np
import os
import torch
from skimage import io, transform
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class ShipDataset(Dataset):
    def __init__(self, ImagePath, LabelPath, augment=None):
        # All this list stored direction of Img.
        self.images = np.array([x.path for x in os.scandir(ImagePath)
                                if x.name.endswith('image.mhd')])
        self.labels = np.array([x.path for x in os.scandir(LabelPath)
                                if x.name.endswith('label.mhd')])
        self.augment = augment  # Need augment the Image?

    def __getitem__(self, index):
        # Load the Image and return it.
        if self.augment:
            image = io.imread(self.images[index])
            image = self.augment(image)
            label = io.imread(self.labels[index])
            return torch.Tensor(image), torch.Tensor(label)

    def __len__(self):
        # The number of Image returned
        return len(self.images),len(self.labels)
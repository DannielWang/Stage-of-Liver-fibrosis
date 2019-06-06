from torch.utils.data import Dataset
import numpy as np
import os
import torch
from skimage import io, transform
import SimpleITK as sitk


class ShipDataset(Dataset):
    def __init__(self, ImagePath, LabelPath, augment=None):
        # All this list stored direction of Img.
        self.image_paths = np.array([x.path for x in os.scandir(ImagePath)
                                     if x.name.endswith('image.mhd')])
        self.label_paths = np.array([x.path for x in os.scandir(LabelPath)
                                     if x.name.endswith('label.mhd')])
        self.augment = augment  # Need augment the Image?

        self.images = []
        self.labels = []
        for i in range(len(self.image_paths)):
            image = sitk.ReadImage(self.image_paths[i])
            label = sitk.ReadImage(self.label_paths[i])
            image = sitk.GetArrayFromImage(image)
            for j in range(image.shape[0]):
                self.images.append(image[j])
                self.labels.append(label[j])

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
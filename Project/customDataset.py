from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import datasets, transforms
import torch
import numpy as np

slicing_coef = 1


class EmotionsDataset(Dataset):
    # we are going to load all images here from the folders
    # we need an array with images, and another one with labels
    # we are gonna store images as tensors
    def __init__(self, datapath):
        self.transforms = transforms.ToTensor()
        self.data = []
        self.targets = []


        for classIndex, folder in enumerate(os.listdir(datapath)):
            folderpath = os.path.join(datapath, folder)

            print("loading folder...")

            reducedimagenamelist = os.listdir(folderpath)[::slicing_coef]

            for image in reducedimagenamelist:
                imagepath = os.path.join(folderpath, image)
                # The images are already in grayscale. Each pixel is a uint8 (8bit number)
                tensor = self.transforms(Image.open(imagepath))
                self.data.append(tensor)
                self.targets.append(torch.tensor(classIndex))


    def __len__(self):
        return len(self.data)

    # we want this to return an image and its label
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    



from __future__ import print_function, division

import os
from os.path import join
import numpy as np
import torch
import pandas as pd
import skimage
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# plt.ion()   # interactive mode

ORIG_IMG_HEIGHT = 480 # These are the dimensions used as input for the ConvNet architecture, so these are independent of actual image size
ORIG_IMG_WIDTH = 640

def show_position(image, label):
    """Shows sample image with ground truth label
    
    Parameters
    ----------
    image : array (height, width, 3)
        RGB image
    label : (2, x, y)
        ground truth x, y location of left and right fingertip (label)
    
    """

    plt.imshow(image)
    plt.scatter(label[0], label[1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


class EndEffectorPositionDataset(Dataset):
    """End effector finger tip dataset."""

    def __init__(self, root_dir, transform=None, load_data_and_labels_from_same_folder=False, use_cuda=True):
        """Initializes dataset
        
        Parameters
        ----------
        root_dir : str
            path to raw data
        transform : list, optional
            list of transforms (check torchsample subfolder for all available transforms, or use official torch.transforms library) (the default is None, which does not load any transforms)
        load_data_and_labels_from_same_folder : bool, optional
            checks whether separate folders for images and labels exist, or whether both are in same folder (the default is False, which assumes a separate 'images' and 'labels' folder exist in root_dir) 
        use_cuda : bool, optional
            whether or not to use GPU (the default is True, which uses GPU)
        
        """

        self.root_dir = root_dir
        self.transform = transform
        self.load_data_and_labels_from_same_folder = load_data_and_labels_from_same_folder

    def __len__(self):
        """Computes length of dataset (number of overall samples)
        
        Returns
        -------
        int 
            number of all samples in dataset
        """

        if not self.load_data_and_labels_from_same_folder:
            return len(os.listdir(join(self.root_dir, 'images')))
        else:
            return len([f for f in os.listdir(self.root_dir) if f.endswith('.png') or f.endswith('.jpg')])
    
    def __getitem__(self, idx):
        """Loads an item from the dataset
        
        Parameters
        ----------
        idx : int
            index of current sample
        
        Returns
        -------
        dict
            Returns a datasample dict containing the keys 'image' and 'label', where label has the same dimension as 'image', but 2 channels, one for each finger tip
        """

        if not self.load_data_and_labels_from_same_folder:
            img_name = join(self.root_dir,
                                    'images', '{0:06d}.png'.format(idx))
            image = io.imread(img_name)
            if image.shape[-1] == 4:
                image = image[:, :, :-1]
            #image = np.transpose(image, (2, 0, 1)) # check whether correct transpose (channels first vs channels last)
            label = np.load(join(self.root_dir, 'labels', '{0:06d}.npy'.format(idx)))[..., :2] # only x and y needed
            label = np.round(label).astype(np.int32)
        else:
            img_name = join(self.root_dir,
                                '{0:06d}.png'.format(idx))
            image = io.imread(img_name)
            if image.shape[-1] == 4:
                image = image[:, :, :-1]
            #image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            label = np.load(join(self.root_dir, 
                                '{0:06d}.npy'.format(idx)))[..., :2] # only x and y needed
            label = np.round(label).astype(np.int32)            
            
        label_l = label[0] # left ee tip
        label_r = label[1] # right ee tip
        buff = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.int64)
        buff[label_l[1], label_l[0], 0] = 1
        buff[label_r[1], label_r[0], 1] = 1

        label = buff
        image = skimage.img_as_float32(image)
        sample = {'image': image, 'label': label}

        # Transform image if provided
        if self.transform:
            sample = self.transform(sample)
        return sample

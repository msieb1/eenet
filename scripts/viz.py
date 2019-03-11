import os, argparse, logging
import sys
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import skimage
import pickle
import io

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils

from tqdm import trange, tqdm
from ipdb import set_trace as st
from models.eenet import define_model
from util.utils import weight_init, set_gpu_mode, zeros, get_numpy
from util.eebuilder import EndEffectorPositionDataset, ConcatDataset, Subset, Transform, FixIndices
from util.transforms import Rescale, RandomRot, RandomCrop, ToTensor, RandomVFlip, RandomHFlip, RandomBlur, RandomSquare, RandomVLines, RandomScaledCrop
from torchsample.transforms.affine_transforms import Rotate, RotateWithLabel, RandomChoiceRotateWithLabel

from pdb import set_trace as st
from simple_graphs import plot_losses
from tensorboardX import SummaryWriter

### Set GPU visibility
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]= "1, 2"  # Set this for adequate GPU usage

### Set Global Parameters
_LOSS = nn.NLLLoss
# ROOT_DIR = '/home/msieb/projects/bullet-demonstrations/experiments/reach/data'
ROOT_DIR = '/home/ahuang/git/eenet/'
IMG_HEIGHT = 240 # These are the dimensions used as input for the ConvNet architecture, so these are independent of actual image size
IMG_WIDTH = 320

np.random.seed()



def visualize(sample): 
    image = sample['image'].numpy().transpose((1, 2, 0))
    label = sample['label'].numpy().transpose((1, 2, 0))

    plt.figure() 
    plt.imshow(image)
    for i in range(np.shape(label)[-1]): 
        point = list(np.where(label[:, :, i] == 1))
        if len(point[0]) != 0: 
            plt.scatter(point[1], point[0], color='red')
    plt.show()

def data_to_tensor(data): 
    tensor = Transform(data, transforms.Compose([Rescale((IMG_HEIGHT, IMG_WIDTH)), ToTensor()]))
    return tensor

def delete_incomplete_data(dataset):
    n = len(dataset)
    valid_indices = []
    for i in range(n): 
        if dataset[i] != None: 
            valid_indices.append(i)
    
    cleaned_dataset = Subset(dataset, valid_indices)
    return cleaned_dataset


if __name__ == '__main__':
    """Parses arguments, creates dataloaders for training and test data, sets up model and logger, and trains network
    """

    ### Setting up parser, logger and GPU params
    set_gpu_mode(True)
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', '-t', type=float, default=0.2)
    parser.add_argument('--epochs', '-e', type=int, default=31)
    parser.add_argument('--learning_rate', '-r', type=float, default=1e-4)
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--load_model', type=bool, default=False)
    # parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--root_dir', type=str, default='')
    # parser.add_argument('-test_files', '--list', nargs='+', help='<Required> Set flag', required=False)
    parser.add_argument('-data_files', '--list', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('-sf', '--load_data_and_labels_from_same_folder', action='store_true')
    parser.add_argument('-sd', '--split_data', action='store_true', default=True)
    args = parser.parse_args()



    
    logging.info('Make sure you provided the correct GPU visibility in line 24 depending on your system !')
    logging.info('Loading {}'.format(args.root_dir))
    logging.info('Processing Data')

    

    dataset_names = args.list

    print('dataset : ', dataset_names[0])



    dataset_tr = EndEffectorPositionDataset(root_dir=args.root_dir + dataset_names[0],                                       
                                        load_data_and_labels_from_same_folder=args.load_data_and_labels_from_same_folder)
   

    dataset_tr = delete_incomplete_data(dataset_tr)
    
    print('training length: ', len(dataset_tr))
   
    plot_img_tr = Subset(dataset_tr, np.random.choice(range(len(dataset_tr)), size=args.n, replace=False))

    
    data = data_to_tensor(plot_img_tr)

    n = len(data)
    for i in range(n): 
    	visualize(data[i])


    # n = len(augmented)
    # for i in range(n): 
    #     visualize(augmented[i])
    #     if i > 10:
    #         break

    # import ipdb; ipdb.set_trace()
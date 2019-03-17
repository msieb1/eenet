import os, argparse, logging
import sys
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import skimage
import pickle
import io
import random 

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

def create_model(args, use_cuda=True):
    """Creates neural network model, loading from checkpoint of provided
    
    Parameters
    ----------
    args : variable function arguments
        see parser in main for details
    use_cuda : bool, optional
        whether or not to use GPU (the default is True, which uses GPU)
    
    Returns
    -------
    torch.nn.Module
        contains EENet model
    """

    model = define_model(IMG_HEIGHT, IMG_WIDTH, use_cuda)
    # tcn = PosNet()
    if args.model_path != '':
        model_path = os.path.join(
            args.model_path,
        )
        # map_location allows us to load models trained on cuda to cpu.
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    if use_cuda:
        model = model.cuda()
    return model

# def visualize(sample): 
#     image = sample['image'].numpy().transpose((1, 2, 0))
#     label = sample['label'].numpy().transpose((1, 2, 0))

#     plt.figure() 
#     plt.imshow(image)
#     for i in range(np.shape(label)[-1]): 
#         point = list(np.where(label[:, :, i] == 1))
#         if len(point[0]) != 0: 
#             plt.scatter(point[1], point[0], color='red')
#     plt.show()
#     plt.close()

def delete_incomplete_data(dataset):
    n = len(dataset)
    valid_indices = []
    for i in range(n): 
        if dataset[i] != None: 
            valid_indices.append(i)
    
    cleaned_dataset = Subset(dataset, valid_indices)
    return cleaned_dataset

def data_to_tensor(data): 
    tensor = Transform(data, transforms.Compose([Rescale((IMG_HEIGHT, IMG_WIDTH)), ToTensor()]))
    return tensor


def visualize(image, label, pred): 
    fig = plt.figure()
    plt.imshow(np.transpose(np.squeeze(image.astype(np.uint8)), (1, 2, 0)))
    colors=['r', 'b']
    plt.scatter(pred[0, 0], pred[0, 1], color='b', s=2)
    plt.scatter(pred[1, 0], pred[1, 1], color='r', s=2)
    plt.scatter(label[0, 0], label[0, 1], color='b', s=2, marker='^')
    plt.scatter(label[1, 0], label[1, 1], color='r', s=2, marker='^')

    return fig

def get_labels(label): 
    buf_l = np.where(label[:, 0, ...].cpu().numpy() ==1)[1:]
    buf_r = np.where(label[:, 1, ...].cpu().numpy() ==1)[1:]
    
    label = [(buf_l[0][0], buf_l[1][0]), (buf_r[0][0], buf_r[1][0])]
    
    return np.asarray(label)

def get_pred_labels(pred): 
    pred_l = pred[..., 0]
    pred_r = pred[..., 1]

    pred_l /= np.abs(np.sum(np.exp(pred_l)))
    pred_r /= np.abs(np.sum(np.exp(pred_r)))

    pred_label_l = np.where(pred_l >= np.max(pred_l))[1:]
    pred_label_r = np.where(pred_r >= np.max(pred_r))[1:]

    pred_point = [[pred_label_l[0][0], pred_label_l[1][0]], [pred_label_r[0][0], pred_label_r[1][0]]]

    return np.asarray(pred_point)

def imshow_heatmap(img, pred):
    """Displays the a prediction heatmap, the max of the heatmap, and the ground truth label
    
    Parameters
    ----------
    img : array (height, width, 3)
        Raw input RGB image
    pred : array (height, width, 1)
        heatmap over image in logits (unnormalized probabilities)
    """
    fig = plt.figure()

    plt.imshow(np.transpose(np.squeeze(img.astype(np.uint8)), (1, 2, 0)))

    pred_l = pred[..., 0]
    pred_r = pred[..., 1]

    pred_l /= np.abs(np.sum(np.exp(pred_l)))
    pred_r /= np.abs(np.sum(np.exp(pred_r)))

    pred_label_l = np.where(pred_l >= np.max(pred_l))[1:]
    pred_label_r = np.where(pred_r >= np.max(pred_r))[1:]

    plt.scatter(pred_label_l[1], pred_label_l[0], s=50, marker='.', c='r')
    plt.scatter(pred_label_r[1], pred_label_r[0], s=50, marker='.', c='b')


    pred_combined_heatmap = pred_l + pred_r
    pred_combined_heatmap /= np.abs(np.sum((pred_combined_heatmap)))
    plt.imshow(np.squeeze(pred_combined_heatmap), cmap="YlGnBu", interpolation='bilinear', alpha=0.3)
    return fig 


if __name__ == '__main__':

    set_gpu_mode(True)
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('-sf', '--load_data_and_labels_from_same_folder', action='store_true')
    parser.add_argument('--epochs', '-e', type=int, default=20)
    parser.add_argument('--learning_rate', '-r', type=float, default=1e-4)
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    args = parser.parse_args()

    out_path = './logs/test'

    if not os.path.exists(out_path + '/tensorboard/'):
        os.makedirs(out_path + '/tensorboard/')

    writer = SummaryWriter(out_path + '/tensorboard/')

    model = create_model(args)
    
    dataset = EndEffectorPositionDataset(root_dir=args.data_path,                                       
                                        load_data_and_labels_from_same_folder=args.load_data_and_labels_from_same_folder)

    dataset = data_to_tensor(delete_incomplete_data(dataset))

    n = len(dataset)
    use_cuda = True

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    i = 0
    for sample in tqdm(loader, leave=False, desc='Eval'):
        xb = sample['image']
        yb = sample['label']
            
        if use_cuda:
            xb = xb.cuda()
            yb = yb.cuda()
        
        pred = model(xb)
        label_point = get_labels(yb)
        pred_point = get_pred_labels(pred.detach().cpu().numpy())
        acc_val = np.linalg.norm(label_point - pred_point)
        print('image ' + str(i) + ' | l2 distance: ', acc_val)
        i+=1

        fig = imshow_heatmap(xb.detach().cpu().numpy(), pred)

        writer.add_figure('TESTING', fig, global_step=i)
            
        if i > 10: 
            break





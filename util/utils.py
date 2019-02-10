import numpy as np
from sklearn.model_selection import train_test_split
from os.path import join
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pdb import set_trace as st


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


"""
GPU wrappers from 
https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/pytorch_util.py
"""

_use_gpu = False
device = None

def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")

def gpu_enabled():
    return _use_gpu

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    return torch.FloatTensor(*args, **kwargs).to(device)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def get_numpy(tensor):
    # not sure if I should do detach or not here
    return tensor.to('cpu').detach().numpy()

def zeros(*sizes, **kwargs):
    return torch.zeros(*sizes, **kwargs).to(device)

def ones(*sizes, **kwargs):
    return torch.ones(*sizes, **kwargs).to(device)

def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs).to(device)

def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs).to(device)

def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)
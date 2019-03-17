import argparse
import torch
import skimage
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from os.path import join
from tensorboardX import SummaryWriter


import ipdb; ipdb.set_trace()
from models.eenet import define_model
from util.transforms import Rescale, ToTensor 

IMG_HEIGHT = 240 # These are the dimensions used as input for the ConvNet architecture, so these are independent of actual image size
IMG_WIDTH = 320

def apply(func, M):
    """Applies a function over a batch (PyTorch as of now has no support for arbitrary function calls on batches)
    """

    tList = [func(m) for m in torch.unbind(M, dim=0) ]
    res = torch.stack(tList, dim=0)
    return res

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

def imshow_heatmap(img, pred):
    """Displays the a prediction heatmap, the max of the heatmap, and the ground truth label
    
    Parameters
    ----------
    img : array (height, width, 3)
        Raw input RGB image
    pred : array (height, width, 1)
        heatmap over image in logits (unnormalized probabilities)
    """

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

def show_heatmap_of_samples(model, images, path, use_cuda=True):
    """Runs image through model and call imshow_heatmap to plot results
    
    Parameters    

    ----------
    dataiter : dataloader as iterator
        Used to obtain some test samples to plot
    model : torch.nn.Module
        neural network model
    use_cuda : bool, optional
        whether or not to use GPU (the default is True, which uses GPU)
    
    """
    tsfm_rescale = Rescale((IMG_HEIGHT, IMG_WIDTH))
    N = len(images)

    model.eval()
    print('write path: ', './logs/' + path + '/tensorboard/')
    writer = SummaryWriter('./logs/' + path + '/tensorboard/')
    
    for i in range(N):
        image = images[i]

        # tsfm_img = tsfm_rescale(image)
        tsfm_img = skimage.transform.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        tsfm_img = tsfm_img.transpose((2, 0, 1))
        tsfm_img = torch.from_numpy(tsfm_img).float().unsqueeze(0)        

        if use_cuda:
            tsfm_img = tsfm_img.cuda()

        fig = plt.figure()
        pred = model(tsfm_img)
        imshow_heatmap(skimage.img_as_ubyte(tsfm_img.cpu().numpy()), pred.cpu().detach().numpy())
        plt.savefig('./logs/' + path + '/{0:05d}.jpg'.format(i))
        writer.add_figure('TESTING', fig, global_step=i)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='')
    args = parser.parse_args()

    ### Load model
    print(args.model_path)
    model = create_model(args)
    
    #path = '/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/gps_runs/bottle_in_mug/videos/train/success_2_view0.mp4'
    path = './success_2_view0.mp4'
    reader = imageio.get_reader(path)
    
    images = []
    reader = list(reader)[-400:]
    for img in reader:
        images.append(img)

    print('./logs/' + args.out_dir)
    


    show_heatmap_of_samples(model, images, args.out_dir)

    import ipdb; ipdb.set_trace()
    exit()
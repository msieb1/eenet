import os, argparse, logging
import sys
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import skimage
import pickle

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
from util.eebuilder import EndEffectorPositionDataset, ConcatDataset, Subset, Transform
from util.transforms import Rescale, RandomRot, RandomCrop, ToTensor, RandomVFlip, RandomHFlip
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
ROOT_DIR = '/home/ahuang/git/eenet/1_view0_logs'
IMG_HEIGHT = 240 # These are the dimensions used as input for the ConvNet architecture, so these are independent of actual image size
IMG_WIDTH = 320

np.random.seed()

### Helper functions
def apply(func, M):
    """Applies a function over a batch (PyTorch as of now has no support for arbitrary function calls on batches)
    """

    tList = [func(m) for m in torch.unbind(M, dim=0) ]
    res = torch.stack(tList, dim=0)
    return res


def imshow_heatmap(img, pred, label):
    """Displays the a prediction heatmap, the max of the heatmap, and the ground truth label
    
    Parameters
    ----------
    img : array (height, width, 3)
        Raw input RGB image
    pred : array (height, width, 1)
        heatmap over image in logits (unnormalized probabilities)
    label : (2, x, y)
        Ground truth finger tip locations
    
    """

    # img = img / 2 + 0.5     # unnormalize?
    plt.imshow(np.transpose(np.squeeze(img.astype(np.uint8)), (1, 2, 0)))

    plt.scatter(label[0][1], label[0][0], s=10, marker='^', c='r', alpha=0.7) # plot left fingertip
    plt.scatter(label[1][1], label[1][0], s=10, marker='^', c='b', alpha=0.7) # plot right fingertip


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

def show_heatmap_of_samples(samples, model, iter, use_cuda=True):
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

    n_display = 8
    for i in range(n_display):
        plt.subplot(2, 4, i+1)
        sample = samples[i] # samples[np.random.choice[np.arange(len(samples), dtype=np.uint8)]]
        image = sample['image']
        label = sample['label']
        if use_cuda:
            image = image.cuda()
            label = label.cuda()
        buf_l = np.where(label[:, 0, ...].cpu().numpy() ==1)[1:]
        buf_r = np.where(label[:, 1, ...].cpu().numpy() ==1)[1:]
        
        if len(buf_l) == 2 and len(buf_r) == 2: 
            label = [(buf_l[0][0], buf_l[1][0]), (buf_r[0][0], buf_r[1][0])]
        elif len(buf_r) < 2: 
            label = [(buf_l[0][0], buf_l[1][0])]
        else: 
            label = [(buf_r[0][0], buf_r[1][0])]
        model.eval()
        pred = model(image.cuda())
        model.train()
        imshow_heatmap(skimage.img_as_ubyte(image.cpu().detach().numpy()), pred.cpu().detach().numpy(), label)
    plt.savefig('./heatmaps/iter_' + str(iter) + '.png')
    plt.close()


def l2_distance(pred, label): 

    pred_l = pred[..., 0].detach().cpu().numpy()
    pred_r = pred[..., 1].detach().cpu().numpy()

    pred_l /= np.abs(np.sum(np.exp(pred_l)))
    pred_r /= np.abs(np.sum(np.exp(pred_r)))

    pred_label_l = np.where(pred_l >= np.max(pred_l))[1:]
    pred_label_r = np.where(pred_r >= np.max(pred_r))[1:]


    yb_l = label[:, 0 ,...].detach().cpu().numpy()
    yb_r = label[:, 1 ,...].detach().cpu().numpy()

    label_l = np.where(yb_l == 1)[1:]
    label_r = np.where(yb_r == 1)[1:]


    pred_point = np.asarray([[np.mean(pred_label_l[0]), np.mean([pred_label_l[1]])], [np.mean(pred_label_r[0]), np.mean([pred_label_r[1]])]]).astype(int)
    label_point = np.asarray([[label_l[0][0], label_l[1][0]], [label_r[0][0], label_r[1][0]]])

    l2_error = np.linalg.norm(pred_point - label_point) 
        
    return l2_error


def train(model, loader_tr, loader_val, loader_t, lr=1e-4, epochs=1000, use_cuda=True):
    """Train model and shows sample results
    
    Parameters
    ----------
    model : torch.nn.Module
        neural network
    loader_tr : Training Dataloader
        Loads training data (images and labels)
    loader_t : Test Dataloader
        Loads test data (images and labels)
    lr : float optional
        Optimizer learning rate (the default is 1e-4)
    epochs : int, optional
        number of training epochs (the default is 1000)
    use_cuda : bool, optional
        whether or not to use GPU (the default is True, which uses GPU)
    
    Returns
    -------
    dictionary of training statistics
        contains metrics such as loss and accuracy
    """

    logdir = ROOT_DIR + '_logs/'
    writer = SummaryWriter(logdir)
    print('writing logs to: ', logdir)

    logs = {
        'loss': {
            'tr': [],
            'val': [], 
            't': []
        },
        'acc': {
            'tr': [],
            'val': [], 
            't': []
        }
    }
    criterion = _LOSS()
    opt = optim.Adam(model.parameters(), lr=lr)
    t_epochs = trange(epochs, desc='{}/{}'.format(0, epochs))
    num_batches_tr = len(loader_tr)
    num_batches_t = len(loader_t)
    num_batches_val = len(loader_val)
    dataiter_t = list(iter(loader_t))
    dataiter_val = list(iter(loader_val))

    for e in t_epochs:
        # Train
        loss_tr = 0
        acc_tr = 0
        t_batches = tqdm(loader_tr, leave=False, desc='Train')
        # show heatmap of samples
        if (e % 9 == 0):
            show_heatmap_of_samples(dataiter_val[:10], model, e)
        
        for sample in t_batches:
            xb = sample['image']
            yb = sample['label']
            if use_cuda:
                xb = xb.cuda()
                yb = yb.cuda()
            opt.zero_grad()

            pred = model(xb)
            pred_flattened = pred.view(pred.size()[0], -1, 2)
            pred_l = pred_flattened[..., 0]
            pred_r = pred_flattened[..., 1]

            yb_flattened = yb.view(yb.size()[0], 2, -1)
            yb_l = yb_flattened[:, 0 ,...]
            yb_r = yb_flattened[:, 1 ,...]


            loss_l = criterion(pred_l, torch.max(yb_l, 1)[1])
            loss_r = criterion(pred_r, torch.max(yb_r, 1)[1])
            loss = loss_l + loss_r
            # loss = criterion(pred, torch.max(yb.view(yb.size()[0], -1), 1)[1])

            acc = l2_distance(pred, yb)
            loss_tr += loss
            acc_tr += acc

            loss.backward()
            opt.step()

            # t_batches.set_description('Train: {:.2f}, {:.2f}'.format(loss, acc))
            t_batches.update()
        

        loss_tr /= num_batches_tr
        acc_tr /= num_batches_tr

        
        # import ipdb; ipdb.set_trace()  


        ## TODO implement validation
        loss_val = 0
        # acc_val = test(model, dataiter_val)
        acc_val = 0
        for sample in tqdm(loader_val, leave=False, desc='Eval'):
            xb = sample['image']
            yb = sample['label']
            if use_cuda:
                xb = xb.cuda()
                yb = yb.cuda()
            opt.zero_grad()
            with torch.no_grad():
                pred = model(xb)
                pred_flattened = pred.view(pred.size()[0], -1, 2)
                pred_l = pred_flattened[..., 0]
                pred_r = pred_flattened[..., 1]

                yb_flattened = yb.view(yb.size()[0], 2, -1)
                yb_l = yb_flattened[:, 0 ,...]
                yb_r = yb_flattened[:, 1 ,...]

                loss_l = criterion(pred_l, torch.max(yb_l, 1)[1])
                loss_r = criterion(pred_r, torch.max(yb_r, 1)[1])
                loss = loss_l + loss_r
                loss_val += loss

                label_point = get_labels(yb)
                pred_point = get_pred_labels(pred.detach().cpu().numpy())
                acc_val += np.linalg.norm(label_point - pred_point)

        loss_val /= num_batches_val
        acc_val /= num_batches_val


        # Eval on test
        loss_t = 0
        acc_t = 0
        for sample in tqdm(loader_t, leave=False, desc='Test'):
            xb = sample['image']
            yb = sample['label']
            if use_cuda:
                xb = xb.cuda()
                yb = yb.cuda()
            opt.zero_grad()
            with torch.no_grad():
                pred = model(xb)
                pred_flattened = pred.view(pred.size()[0], -1, 2)
                pred_l = pred_flattened[..., 0]
                pred_r = pred_flattened[..., 1]

                yb_flattened = yb.view(yb.size()[0], 2, -1)
                yb_l = yb_flattened[:, 0 ,...]
                yb_r = yb_flattened[:, 1 ,...]

                loss_l = criterion(pred_l, torch.max(yb_l, 1)[1])
                loss_r = criterion(pred_r, torch.max(yb_r, 1)[1])
                loss = loss_l + loss_r
                loss_t += loss

                label_point = get_labels(yb)
                pred_point = get_pred_labels(pred.detach().cpu().numpy())
                acc_t += np.linalg.norm(label_point - pred_point)


        loss_t /= num_batches_t
        acc_t /= num_batches_t

        writer.add_scalar('data/train_loss', loss_tr, e)
        writer.add_scalar('data/train_acc', acc_tr, e)

        
        t_epochs.set_description('{}/{} | Tr {:.2f}, {:.2f}. T {:.2f}, {:.2f}'.format(e, epochs, loss_tr, acc_tr, loss_t, acc_t))
        t_epochs.update()
        print('epoch: ', e)
        print('train_loss: ', loss_tr)
        print('val_loss: ', loss_val)
        print('test_loss: ', loss_t)
        logs['loss']['tr'].append(loss_tr)
        logs['acc']['tr'].append(acc_tr)
        logs['loss']['val'].append(loss_val)
        logs['acc']['val'].append(acc_val)
        logs['loss']['t'].append(loss_t)
        logs['acc']['t'].append(acc_t)
        print('-'*10)
    return logs

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
    if args.load_model:
        model_path = os.path.join(
            args.model_path,
        )
        # map_location allows us to load models trained on cuda to cpu.
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    if use_cuda:
        model = model.cuda()
    return model

def visualize(sample, j): 
    image = sample['image'].numpy().transpose((1, 2, 0))
    label = sample['label'].numpy().transpose((1, 2, 0))

    plt.figure() 
    plt.imshow(image)
    for i in range(np.shape(label)[-1]): 
        point = list(np.where(label[:, :, i] == 1))
        if len(point[0]) != 0: 
            plt.scatter(point[1], point[0], color='red')
    # plt.savefig('./0_view1/' + str(j) + '.png')
    plt.show()

def delete_incomplete_data(dataset):
    n = len(dataset)
    valid_indices = []
    for i in range(n): 
        if dataset[i] != None: 
            valid_indices.append(i)
    
    cleaned_dataset = Subset(dataset, valid_indices)
    return cleaned_dataset

def get_pred_labels(pred): 
    pred_l = pred[..., 0]
    pred_r = pred[..., 1]

    pred_l /= np.abs(np.sum(np.exp(pred_l)))
    pred_r /= np.abs(np.sum(np.exp(pred_r)))

    pred_label_l = np.where(pred_l >= np.max(pred_l))[1:]
    pred_label_r = np.where(pred_r >= np.max(pred_r))[1:]

    pred_point = [[pred_label_l[0][0], pred_label_l[1][0]], [pred_label_r[0][0], pred_label_r[1][0]]]

    return np.asarray(pred_point)

def get_labels(label): 
    buf_l = np.where(label[:, 0, ...].cpu().numpy() ==1)[1:]
    buf_r = np.where(label[:, 1, ...].cpu().numpy() ==1)[1:]
    
    if len(buf_l) == 2 and len(buf_r) == 2: 
        label = [(buf_l[0][0], buf_l[1][0]), (buf_r[0][0], buf_r[1][0])]
    elif len(buf_r) < 2: 
        label = [(buf_l[0][0], buf_l[1][0])]
    else: 
        label = [(buf_r[0][0], buf_r[1][0])]
    return np.asarray(label)

def test(model, samples, use_cuda=True): 
    n = len(samples)
    print('test n: ', n)
    l2_error = 0
    for i in range(n): 
        sample = samples[i] 
        image = sample['image']
        label = sample['label']
        if use_cuda:
            image = image.cuda()
            label = label.cuda()
        label_point = get_labels(label)

        model.eval()
        pred = model(image.cuda())
        model.train()
        pred_point = get_pred_labels(pred.detach().cpu().numpy())        

        l2_error += np.linalg.norm(label_point - pred_point)
    
    return l2_error / float(n)




if __name__ == '__main__':
    """Parses arguments, creates dataloaders for training and test data, sets up model and logger, and trains network
    """

    ### Setting up parser, logger and GPU params
    set_gpu_mode(True)
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', '-t', type=float, default=0.2)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--learning_rate', '-r', type=float, default=1e-4)
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--root_dir', type=str, default=ROOT_DIR)
    parser.add_argument('-sf', '--load_data_and_labels_from_same_folder', action='store_true')
    args = parser.parse_args()


    print('\n')
    logging.info('Make sure you provided the correct GPU visibility in line 24 depending on your system !')
    logging.info('Loading {}'.format(args.root_dir))
    logging.info('Processing Data')

    

    
    ### Create dataset
    dataset = EndEffectorPositionDataset(root_dir=args.root_dir, 
                                        # transform=transforms.Compose(
                                        #     [
                                        #     RandomCrop((400, 400)), 
                                        #     RandomRot(-30., 30.), 
                                        #     Rescale((IMG_HEIGHT, IMG_WIDTH)),
                                        #     ToTensor(),
                                        #     #RandomChoiceRotateWithLabel([0,  177,179,180])
                                        #     ]),                                        
                                        load_data_and_labels_from_same_folder=args.load_data_and_labels_from_same_folder)
    
    dataset = delete_incomplete_data(dataset)
    n = len(dataset)


    n_test = int(n * .15 )  # number of test/val elements
    n_train = n - 2 * n_test
    dataset_tr, dataset_t, dataset_val = random_split(dataset, (n_train, n_test, n_test))
    print(n, n_train, n_test)



    crop_list = [(200, 400), (300, 400), (300, 300), (400, 400), (400, 300), (400, 200)]
    crop_list = [(300, 400), (400, 300)]
    # crop_list = []
    dataset_0 = Transform(dataset_tr, transforms.Compose(
                        [
                        Rescale((IMG_HEIGHT, IMG_WIDTH)),
                        ]))
    for i in range(len(crop_list)): 
        dataset_cropped = Transform(dataset_tr, transforms.Compose(
                        [
                        RandomCrop(crop_list[i]), 
                        ]))
        dataset_0 = ConcatDataset([dataset_0, dataset_cropped])

    dataset_tr = Transform(dataset_0, transforms.Compose(
                        [
                        RandomVFlip(0.5), 
                        RandomHFlip(0.5), 
                        RandomRot(-30., 30., 0.5), 
                        Rescale((IMG_HEIGHT, IMG_WIDTH)),
                        ToTensor()
                        ]))
    print('training size: ', len(dataset_tr))
    
    dataset_t = Transform(dataset_t, transforms.Compose([Rescale((IMG_HEIGHT, IMG_WIDTH)), ToTensor()]))
    dataset_val = Transform(dataset_val, transforms.Compose([Rescale((IMG_HEIGHT, IMG_WIDTH)), ToTensor()]))

    # for i in range(n_train): 
    #     sample = dataset_tr[i]
    #     visualize(sample, i)
    #     if i > 10: 
    #         break

    
    loader_tr = DataLoader(dataset_tr, batch_size=2,
                        shuffle=True, num_workers=4)
    loader_val = DataLoader(dataset_val, batch_size=1, shuffle=True) 
    loader_t = DataLoader(dataset_t, batch_size=1, shuffle=True) 


    ### Load model
    model = create_model(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    # model.to(device)
    # import ipdb; ipdb.set_trace()  

    ### Train
    logging.info('Training.')
    logs = train(model, loader_tr, loader_val, loader_t, lr=args.learning_rate, epochs=args.epochs)
    # TODO save stuff

    with open('1_logs.pkl', 'wb') as f: 
        pickle.dump(logs, f)

    dataiter_t = list(iter(loader_t))
    l2_error = test(model, dataiter_t)
    print('l2 error: ', l2_error)

    plot_losses(logs, '1_goodtransform')



    # Default into debug mode if training is completed
    import ipdb; ipdb.set_trace()
    exit()


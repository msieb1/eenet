from skimage import io, transform
import numpy as np
import torch
from ipdb import set_trace as st 
import matplotlib.pyplot as plt

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        
        tsfm_labels = np.zeros((img.shape[0], img.shape[1], 2))
        for i in range(labels.shape[-1]):
            landmarks = labels[..., i]
            points = list(np.where(landmarks == 1))
            points[0] = np.array(points[0] * new_h / h, dtype=np.int32)
            points[1] = np.array(points[1] * new_w / w, dtype=np.int32)
            buff = np.zeros((img.shape[0], img.shape[1]))
            
            try:
                buff[tuple(points)] = 1
                tsfm_labels[..., i] = buff
            except:
                import ipdb; ipdb.set_trace()
        return {'image': img, 'label': tsfm_labels}

def visualize(sample): 
    image = sample['image']
    label = sample['label']
    plt.figure() 
    plt.imshow(image)
    for i in range(np.shape(label)[-1]): 
        point = list(np.where(label[:, :, i] == 1))
        if len(point[0]) != 0: 
            plt.scatter(point[1], point[0], color='red')
    plt.savefig('./pics/' + str(np.random.randint(0, 10000)) + '.png')
    plt.close()

class RandomRot(object): 
    def __init__(self, lower, upper, p, resize=False, center=None):
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert isinstance(p, float)
        self.lower = lower
        self.upper = upper
        self.resize = resize
        self.center = center
        self.p = p


    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.random() < self.p: 
            while True: 
                theta = np.random.uniform(self.lower, self.upper)
                img_rot = transform.rotate(image, theta, resize=self.resize, order=0, preserve_range=True)
                lbl_rot = transform.rotate(label, theta, resize=self.resize, order=0, preserve_range=True)

                
                # print(len(x), len(y))
                # print(x)
                # print(y)
                # print(np.sum(lbl_rot.flatten()))
                # print(np.sum(label.flatten()))
                # print(np.any(lbl_rot[:, :, 0] == 1))
                # print(np.any(lbl_rot[:, :, 1] == 1))


                if np.any(lbl_rot[:, :, 0] == 1) and np.any(lbl_rot[:, :, 1] == 1): 
                    x = np.where(lbl_rot[:, :, 0] == 1)
                    y = np.where(lbl_rot[:, :, 1] == 1)

                    label = np.zeros(np.shape(lbl_rot))
                    label[np.mean(x[0]).astype(int), np.mean(x[1]).astype(int), 0] = 1
                    label[np.mean(y[0]).astype(int), np.mean(y[1]).astype(int), 1] = 1
                    break
            return {'image': img_rot, 'label': label}
        else: 
            return {'image': image, 'label': label}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        while True: 
        
            crop_h = np.random.randint(0, h - new_h)
            crop_w = np.random.randint(0, w - new_w)
            cropped = image[crop_h:crop_h+new_h, crop_w:crop_w+new_w, :]

            labels_cropped = labels[crop_h:crop_h+new_h, crop_w:crop_w+new_w, :]
            if np.sum(labels_cropped) >= np.sum(labels): 
                break
        return {'image': cropped, 'label': labels_cropped}

class RandomVFlip(object): 
    def __init__(self, p):
        assert isinstance(p, float)
        self.p = p
    def __call__(self, sample): 
        image, label = sample['image'], sample['label']
        if np.random.random() < self.p: 
            return {'image': np.flipud(image), 'label': np.flipud(label)}
        return {'image': image, 'label': label}

class RandomHFlip(object): 
    def __init__(self, p):
        assert isinstance(p, float)
        self.p = p
    def __call__(self, sample): 
        image, label = sample['image'], sample['label']
        if np.random.random() < self.p: 
            return {'image': np.fliplr(image), 'label': np.fliplr(label)}
        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        landmarks = landmarks.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(landmarks).float()}

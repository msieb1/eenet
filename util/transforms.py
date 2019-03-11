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
class RandomSquare(object): 
    def __init__(self, max_num = 4):
        self.max_num = max_num
    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        h, w = image.shape[:2]
        num = np.random.randint(0, self.max_num)
        new = image
        for i in range(num): 

            center_h = np.random.randint(0, h)
            center_w = np.random.randint(0, w)

            width = int(np.random.randint(0, h/2) / 2)
            # print(center_w, center_h, width)
            # import ipdb; ipdb.set_trace()

            x_low = center_h - width
            x_high = center_h + width
            y_low = center_w - width
            y_high = center_w + width

            x_low = max(0, x_low)
            y_low = max(0, y_low)
            x_high = min(x_high, h)
            y_high = min(y_high, w)
            new[x_low:x_high, y_low:y_high, :] = 0

        return {'image': new, 'label': labels}

class RandomVLines(object): 
    def __init__(self, max_num = 4):
        self.max_num = max_num
    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        h, w = image.shape[:2]
        num = np.random.randint(0, self.max_num)
        total_mask = np.ones([h, w, 3])
        for i in range(num): 
            mask = np.zeros([h, w, 3])

            center_h = np.random.randint(0, h)
            center_w = np.random.randint(0, w)

            width = int(np.random.randint(5, int(w/20.)) / 2)
            height = int(np.random.randint(int(h/10.), int(h/2.)) / 2) 
            # print(center_w, center_h, width)
            # import ipdb; ipdb.set_trace()

            x_low = center_h - height
            x_high = center_h + height
            y_low = center_w - width
            y_high = center_w + width

            
            x_low = max(0, x_low)
            y_low = max(0, y_low)
            x_high = min(x_high, h)
            y_high = min(y_high, w)
            mask[x_low:x_high, y_low:y_high, :] = 1
            theta = np.random.uniform(-30, 30)
            mask = transform.rotate(mask, theta, resize=False, order=0, preserve_range=True)
            mask = np.abs((1 - mask))
            total_mask = np.multiply(total_mask, mask)
        new = np.multiply(total_mask, image)
        return {'image': new, 'label': labels}


class RandomBlur(object): 
    def __init__(self, p):
        self.p = p


    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        h, w = image.shape[:2]

        l1 = np.where(labels[:, :, 0] == 1)
        l2 = np.where(labels[:, :, 1] == 1)

        distance = np.sqrt((l1[0][0] - l2[0][0])**2 + (l1[1][0] - l2[1][0])**2)
        width = int(np.random.uniform(0, distance))

        blurred = image
        if np.random.random() < self.p:
            x_low = l1[0][0] - width
            x_high = l1[0][0] + width
            y_low = l1[1][0] - width
            y_high = l1[1][0] + width

        else: 
            x_low = l2[0][0] - width
            x_high = l2[0][0] + width
            y_low = l2[1][0] - width
            y_high = l2[1][0] + width
        x_low = max(0, x_low)
        y_low = max(0, y_low)
        x_high = min(x_high, w)
        y_high = min(y_high, h)
        blurred[x_low:x_high, y_low:y_high, :] = 0
        #np.mean(image[x_low:x_high, y_low:y_high, :], axis=(0, 1))
        return {'image': blurred, 'label': labels}



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

class RandomScaledCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, min_scale, max_scale, min_ar, max_ar):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_ar = min_ar
        self.max_ar = max_ar

    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        h, w = image.shape[:2]

        points = list(np.where(labels == 1))
        h_range = abs(points[0][0] - points[1][0])
        w_range = abs(points[0][1] - points[1][1])

        scale_limit = max(h_range / h, w_range / w)
        min_scale = max(self.min_scale, scale_limit)

        scale = np.random.uniform(min_scale + 0.05, self.max_scale)
        ratio = np.random.uniform(self.min_ar, self.max_ar)
        
        new_w = int(scale * w)
        new_h = int(new_w * ratio)

        new_w = min(w, new_w)
        new_h = min(h, new_h)

        # print(scale, ratio, new_w, new_h)

        
        # import ipdb; ipdb.set_trace()


        while True: 
            if new_h == h: 
                crop_h = 0
            else: 
                crop_h = np.random.randint(0, h - new_h)
            if new_w == w: 
                crop_w = 0
            else: 
                crop_w = np.random.randint(0, w - new_w)

        
            # crop_h = np.random.randint(0, h - new_h)
            # crop_w = np.random.randint(0, w - new_w)
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

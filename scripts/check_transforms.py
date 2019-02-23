import os, sys
ROOT_DIR = '/Users/ottery/Documents/eenet/'
sys.path.append(ROOT_DIR)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split
from util.eebuilder import EndEffectorPositionDataset
from skimage import io, transform
from PIL import Image


def RemoveBadLabels(label, size): 
	return

def RandomRescale(sample, output_size): 
	image, labels = sample['image'], sample['label']
	print(np.sum(labels))
	h, w = image.shape[:2]
	if isinstance(output_size, int):
		if h > w:
			new_h, new_w = output_size * h / w, output_size
		else:
			new_h, new_w = output_size, output_size * w / h
	else:
		new_h, new_w = output_size
	new_h, new_w = int(new_h), int(new_w)
	img = transform.resize(image, (new_h, new_w))
	lbl = transform.resize(labels, (new_h, new_w), order=0, preserve_range=True, anti_aliasing=True)
	return {'image': img, 'label': lbl.astype(np.int32)}


def RandomCrop(sample, output_size):
	image, labels = sample['image'], sample['label']
	h, w = image.shape[:2]
	new_h, new_w = output_size
	
	crop_h = np.random.randint(0, h - new_h)
	crop_w = np.random.randint(0, w - new_w)
	cropped = image[crop_h:crop_h+new_h, crop_w:crop_w+new_w, :]

	labels_cropped = labels[crop_h:crop_h+new_h, crop_w:crop_w+new_w, :]
	
	return {'image': cropped, 'label': labels_cropped}

def RandomHFlip(sample, p): 
	image, label = sample['image'], sample['label']
	if np.random.random() < p: 
		return {'image': np.fliplr(image), 'label': np.fliplr(label)}
	return {'image': image, 'label': label}

def RandomVFlip(sample, p): 
	image, label = sample['image'], sample['label']
	if np.random.random() < p: 
		return {'image': np.flipud(image), 'label': np.flipud(label)}
	return {'image': image, 'label': label}

def RandomTranslate(sample, x, y): 
	return

# similaritytransform

def RandomRotate(sample, theta, resize=False, center=None): 
	image, label = sample['image'], sample['label']
	img_rot = transform.rotate(image, theta, resize=resize, order=0, preserve_range=True)
	lbl_rot = transform.rotate(label, theta, resize=resize, order=0, preserve_range=True)
	return {'image': img_rot, 'label': lbl_rot}


def visualize(sample): 
	image = sample['image']
	label = sample['label']
	plt.figure() 
	plt.imshow(image)
	for i in range(np.shape(label)[-1]): 
		point = list(np.where(label[:, :, i] == 1))
		if len(point[0]) != 0: 
			plt.scatter(point[1], point[0], color='red')
	plt.show()


if __name__ == '__main__':
	
	image_path = './0_view0/'
	file_path = '000125'
	labels = np.load(image_path + file_path + '.npy')
	image = cv2.imread(image_path + file_path + '.png')


	# cropped_image, cropped_labels = RandomCrop(image, labels, [200, 300])
	# embed_image, embed_labels = RandomEmbed(cropped_image, cropped_labels, np.shape(image)[:2])

	dataset = EndEffectorPositionDataset(root_dir=ROOT_DIR + '0_view0',  
                                        # transform=transforms.Compose(
                                        #     [
                                        #     Rescale((IMG_HEIGHT, IMG_WIDTH)),
                                        #     ToTensor(),
                                        #     #RandomChoiceRotateWithLabel([0,  177,179,180])
                                        #     ]),                                        
                                        load_data_and_labels_from_same_folder=True)
	h, w = np.shape(dataset[0]['image'])[:2]

	# visualize(dataset[0]['image'], dataset[0]['label'])
	result = RandomCrop(dataset[0], [200, 400])
	visualize(result)

	theta = 15
	resize=False
	center=None

	result2 = RandomRotate(result, theta, resize, center)
	visualize(result2)

	# plt.figure()
	# plt.imshow(cropped)
	# plt.imshow(label[:, :, 0], alpha = 0.2)
	# plt.imshow(label[:, :, 1], alpha = 0.2)
	# plt.show()


	# result2 = RandomRescale(result, [h, w])
	# visualize(result2)
	# label = result2['label']
	# print(list(np.where(label!=0)))



	# plt.figure()
	# plt.subplot(2, 2, 1)
	# plt.imshow(image)
	# plt.scatter(labels[:, 0], labels[:, 1], color=['red'])
	# plt.subplot(2, 2, 3)
	# plt.imshow(cropped_image)
	# plt.scatter(cropped_labels[:, 0], cropped_labels[:, 1], color=['red'])
	# plt.subplot(2, 2, 4)
	# plt.imshow(embed_image)
	# plt.scatter(embed_labels[:, 0], embed_labels[:, 1], color=['red'])
	# plt.show()
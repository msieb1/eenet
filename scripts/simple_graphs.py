import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle 

def plot_all_losses(folders, labels): 
	plt.figure()
	plt.title('All Training Loss')
	for i in range(len(folders)): 
		logs = load_data(folders[i])
		plt.plot(logs['loss']['tr'], label=labels[i])
	plt.legend(loc='upper right')
	plt.savefig('./all_trainingloss.png')
	plt.close()


def plot_tr_loss(logs, folder): 
	plt.figure()
	plt.title(folder + ' Training Loss')
	plt.plot(logs['loss']['tr'])
	plt.ylim((0, 25))
	plt.savefig('./' + folder + '/' + folder + '_trainingloss.png')
	plt.close()

def plot_l2(logs, keys, folder): 
	plt.figure()
	plt.title(folder + ' Ave. L2 Distance')
	for key in keys:
		plt.plot(logs['acc'][key], label=key)
	plt.legend(loc='upper right')
	plt.ylim((0, 200))
	plt.savefig('./' + folder + '/' + folder + '_l2.png')
	plt.close()

def load_data(folder): 
	name = 'logs.pkl'

	with open('./' + folder + '/' + name, 'rb') as f: 
		logs = pickle.load(f)
	return logs

if __name__ == '__main__':
	folders = ['nothing_data', 'nocrop_data', 'crop_data']
	labels = ['no transforms', 'random rot/flip', 'with cropping']
	for folder in folders: 
		logs = load_data(folder)

		keys = ['tr', 'val', 't']
		plot_l2(logs, keys, folder)
	
		plot_tr_loss(logs, folder)
	
	plot_all_losses(folders, labels)




import numpy as np
import os
import math
from os import listdir
from os.path import isfile, join
from scipy.misc import imsave

class nova_set(object):
	'''
	NOVA data set is contained of 2 complex matrices divided to 4 float matrices (2*[real+imag]).
	Each NOVA image has the dimensions of 30 * 30.
	Each NOVA vector data has the dimensions of 8 * 1 (8 features) and will be used to train 
	the generator instead of general noise.
	'''

	def __init__(self, batch_size ,dir_path):
		if not os.path.exists(dir_path):
			raise ValueError('Directory: {} does not exist!'.format(dir_path))
		else:
			files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
			self.mat_trans_files = [join(dir_path, f) for f in files if 't_' in f]
			self.mat_reflc_files = [join(dir_path, f) for f in files if 'r_' in f]
			self.data_file = [join(dir_path, f) for f in files if 'x' in f]
			if len(self.data_file) == 1:
				self.data_file = self.data_file[0]

			self.is_loaded = False
			self.reflactance = dict(zip(['real', 'imag'], [None, None]))
			self.transmitance = dict(zip(['real', 'imag'], [None, None]))
			self.data = None
			self.batch_size = batch_size
			self.batch_counter = 0
			self.num_examples = 0

	def load(self, opt='all'):
		'''
			Load data to cache
			
			Args
				opt - the option (reflactance/transmitance/input data) to be loaded
		'''
		if opt in ['ref', 'all']:
			print('loading ref data...')
			for file_ in self.mat_reflc_files:
				if 'real' in file_:
					self.reflactance['real'] = np.transpose(np.load(file_).astype(dtype=np.float32))
				elif 'imag' in file_:
					self.reflactance['imag'] = np.transpose(np.load(file_).astype(dtype=np.float32))
			self.is_loaded = True
			self.num_examples = self.reflactance['real'].shape[0]

		if opt in ['tra', 'all']:
			print('loading tra data...')
			for file_ in self.mat_trans_files:
				if 'real' in file_:
					self.transmitance['real'] = np.transpose(np.load(file_).astype(dtype=np.float32))
				elif 'imag' in file_:
					self.transmitance['imag'] = np.transpose(np.load(file_).astype(dtype=np.float32))
			
			self.is_loaded = True
			self.num_examples = self.transmitance['real'].shape[0]

		if opt in ['all', 'data']:
			self.data = np.transpose(np.load(self.data_file).astype(dtype=np.float32))
			self.is_loaded = True
			self.num_examples = self.data.shape[0]
	
	


	def next_batch(self, which_config, which_type, batch_size= None):
		'''
			Get permutated data for training procecss
			
			Args
				which_config: chooses reflactance/transmitance returned data 
				which_type	: chooses real/imaginary returned data
		'''
		returned_data = []

		#set the batch boundaries
		begin = self.batch_counter
		if batch_size is None:
			end = self.batch_counter + self.batch_size
			self.batch_counter += self.batch_size
		else:
			end = self.batch_counter + batch_size
			self.batch_counter += batch_size

		#if the batch_counter overflows, set to 0
		if self.batch_counter > self.num_train_examples - 1:
			self.batch_counter = 0
			begin = 0
			if batch_size is None:
				end = self.batch_size
			else:
				end = batch_size
		 

		if 'ref' in which_config:
			returned_data.append(self.perm_reflactance[which_type][begin:end, :])
		
		if 'tra' in which_config:
			returned_data.append(self.perm_transmitance[which_type][begin:end, :])

		if 'data' in which_config:
			returned_data.append(self.perm_data[begin:end, :])

		return returned_data
	


	def get_data(self, which_config, which_type, indices, is_train=True):
		'''
			Get non-permutated data according to indices
			
			Args
				which_config: chooses reflactance/transmitance returned data 
				which_type	: chooses real/imaginary returned data
				indices: chooses the data points to be extracted
				is_train: if true, extracted points will be from the training set, else from the testing set
		'''
		returned_data = []
		if 'ref' in which_config:
			if is_train:
				returned_data.append(self.ref_train[which_type][indices, :])
			else:
				returned_data.append(self.ref_test[which_type][indices, :])

		if 'tra' in which_config:
			if is_train:
				returned_data.append(self.tra_train[which_type][indices, :])
			else:
				returned_data.append(self.tra_test[which_type][indices, :])

		if 'data' in which_config:
			if is_train:
				returned_data.append(self.train_data[indices, :])
			else:
				returned_data.append(self.test_data[indices, :])
		return returned_data


	def save_images(self, which_config, which_type, which_indices, save_dir):
		'''
			Save images to the disk 
			
			Args
				which_config: chooses reflactance/transmitance returned data 
				which_type	: chooses real/imaginary returned data
				which_indices: chooses the data points to be saved
				save_dir 	: the directory where the images will be saved
		'''
		#create the saving directory
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		#get the indexed images
		imgs = None
		if which_config == 'ref':
			imgs = self.ref_train[which_type][which_indices, :]
		else:
			imgs = self.tra_train[which_type][which_indices, :]

		#save the images
		for k in range(len(which_indices)):
			imsave(os.path.join(save_dir, '%d.png') % which_indices[k], imgs[k].reshape(30, 30) )



	def permutate(self):
		'''
			Permutate all the data 
		'''
		self.batch_counter = 0
		self.perm = np.random.permutation(self.num_train_examples)
		if hasattr(self, "ref_train"):
			if all(val is not None for val in self.ref_train.values()):
				values = [value[self.perm, :] for value in self.ref_train.values()]
				self.perm_reflactance = None
				self.perm_reflactance = dict(zip(['real', 'imag'], values))
				del values
		
		if hasattr(self, "tra_train"):
			if all(val is not None for val in self.tra_train.values()):
				values = [value[self.perm, :] for value in self.tra_train.values()]
				self.perm_transmitance = None
				self.perm_transmitance = dict(zip(['real', 'imag'], values))
				del values

		if hasattr(self, "train_data"):
			if self.train_data is not None:
				self.perm_data = self.train_data[self.perm, :]

	

	def num_batches(self):
		'''
			Returns the number of batches 
		'''
		return int(math.floor(self.num_train_examples / self.batch_size))


	def split_data(self, ratio=0.8):
		'''
			Split the data to training and testing data
			Args
				ratio - the ratio according the data will be split to training - testing sets.
		'''
		assert ratio <= 1
		self.num_train_examples = num_trains = int(math.ceil(self.num_examples * ratio))
		
		if all(val is not None for val in self.reflactance.values()):
			train_values = [value[:num_trains, :] for value in self.reflactance.values()]
			test_values  = [value[num_trains:, :] for value in self.reflactance.values()]
			self.ref_train = dict(zip(['real','imag'], train_values))
			self.ref_test  = dict(zip(['real','imag'], test_values))
			del train_values, test_values
			self.reflactance = dict(zip(['real','imag'], [None, None]))

		if all(val is not None for val in self.transmitance.values()):
			train_values = [value[:num_trains, :] for value in self.transmitance.values()]
			test_values  = [value[num_trains:, :] for value in self.transmitance.values()]
			self.tra_train = dict(zip(['real','imag'], train_values))
			self.tra_test  = dict(zip(['real','imag'], test_values))
			del train_values, test_values
			self.transmitance = dict(zip(['real','imag'], [None, None]))

		if self.data is not None:
			self.train_data, self.test_data = self.data[:num_trains, :], self.data[num_trains:, :]
			self.data = None



	def test_info(self, which_config, which_type):
		'''
			Return the test data set
			
			Args
				which_config: chooses reflactance/transmitance returned data 
				which_type	: chooses real/imaginary returned data
		'''
		returned_data = []
		if 'ref' in which_config:
			returned_data.append(self.ref_test[which_type])

		if 'tra' in which_config:
			returned_data.append(self.tra_test[which_type])

		if 'data' in which_config:
			returned_data.append(self.test_data)

		return returned_data

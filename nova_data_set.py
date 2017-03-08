import numpy as np
import os
import math
from os import listdir
from os.path import isfile, join

class nova_set(object):
	'''
	NOVA data set contained of 2 complex matrices divided to 4 float matrices (2*[real+imag]).
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
			self.reflactance  = {'real' : None, 'imag' : None}
			self.transmitance = {'real' : None, 'imag' : None}
			self.data = None
			self.batch_size = batch_size
			self.batch_counter = 0
			self.num_examples = 0

	def load(self, opt='all'):
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
	
		
	def next_batch(self, which_config, which_type):
		returned_data = []

		#set the batch boundaries
		begin = self.batch_counter
		end = self.batch_counter + self.batch_size
		self.batch_counter += self.batch_size

		if self.batch_counter > self.num_examples - 1:
			self.batch_counter = 0
			begin = 0
			end = self.batch_size
		 

		if 'ref' in which_config:
			returned_data.append(self.reflactance[which_type][begin:end, :])
		
		if 'tra' in which_config:
			returned_data.append(self.transmitance[which_type][begin:end, :])

		if 'data' in which_config:
			returned_data.append(self.data[begin:end, :])

		return returned_data

	def permutate(self):
		'''
			Permutate all the data 
		'''
		self.batch_counter = 0
		self.perm = np.random.permutation(self.num_examples)
		if all(val is not None for val in self.reflactance.values()):
			values = [value[self.perm, :] for value in self.reflactance.values()]
			self.reflactance = dict(zip(['real', 'imag'], values))
			del values
		
		if all(val is not None for val in self.transmitance.values()):
			values = [value[self.perm, :] for value in self.transmitance.values()]
			self.transmitance = dict(zip(['real', 'imag'], values))
			del values

		if self.data is not None:
			self.data = self.data[self.perm]


	def num_batches(self):
		return int(math.floor(self.num_examples / self.batch_size))

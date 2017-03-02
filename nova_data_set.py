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
			for file in self.mat_reflc_files:
				if 'real' in file:
					self.reflactance['real'] = np.transpose(np.load(file).astype(dtype=np.float32))
				elif 'imag' in file:
					self.reflactance['imag'] = np.transpose(np.load(file).astype(dtype=np.float32))
			self.is_loaded = True
			self.num_examples = self.reflactance['real'].shape[0]

		if opt in ['tra', 'all']:
			for file in self.mat_trans_files:
				if 'real' in file:
					self.transmitance['real'] = np.transpose(np.load(file).astype(dtype=np.float32))
				elif 'imag' in file:
					self.transmitance['imag'] = np.transpose(np.load(file).astype(dtype=np.float32))
			
			self.is_loaded = True
			self.num_examples = self.transmitance['real'].shape[0]

		if opt in ['all', 'data']:
			self.data = np.transpose(np.load(self.data_file).astype(dtype=np.float32))
			self.is_loaded = True
			self.num_examples = self.data.shape[0]
	
		
	def next_batch(self, which):
		ref_real, tra_real, all_real = None, None, None
		ref_imag, tra_imag = None, None

		#set the batch boundaries
		begin = self.batch_counter
		end = self.batch_counter + self.batch_size
		self.batch_counter += self.batch_size

		if self.batch_counter > self.num_examples - 1:
			self.batch_counter = 0
			begin = 0
			end = self.batch_size
		 

		if which == 'ref':
			try:
				ref_real, ref_imag = self.reflactance['real'][begin:end, :], self.reflactance['imag'][begin:end, :]
			except:
				ref_real, ref_imag = self.reflactance['real'][begin:, :], self.reflactance['imag'][begin:, :]
		
		elif which == 'tra':
			try:
				tra_real, tra_imag = self.transmitance['real'][begin:end, :], self.transmitance['imag'][begin:end, :]
			except:
				tra_real, tra_imag = self.transmitance['real'][begin:, :], self.transmitance['imag'][begin:, :]
		elif which == 'data':
			try:
				all_real = self.data[begin:end, :]
			except:
				all_real = self.data[begin:, :]

		elif which == 'all':
			try:
				ref_real, ref_imag = self.reflactance['real'][begin:end, :], self.reflactance['imag'][begin:end, :]
				tra_real, tra_imag = self.transmitance['real'][begin:end, :], self.transmitance['imag'][begin:end, :]
				all_real = self.data[begin:end, :]
			except:
				ref_real, ref_imag = self.reflactance['real'][begin:, :], self.reflactance['imag'][begin:, :]
				tra_real, tra_imag = self.transmitance['real'][begin:, :], self.transmitance['imag'][begin:, :]
				all_real = self.data[begin:, :]

		return ref_real, ref_imag, tra_real, tra_imag, all_real


	def permutate(self):
		'''
			Permutate all the data 
		'''
		self.batch_counter = 0
		self.perm = np.random.permutation(self.num_examples)
		if all(val is not None for val in self.reflactance.values()):
			values = [value[self.perm, :] for value in self.reflactance.values()]
			self.reflactance = dict(zip(['real', 'imag'], values))
		
		if all(val is not None for val in self.transmitance.values()):
			values = [value[self.perm:, ] for value in self.transmitance.values()]
			self.transmitance = dict(zip(['real', 'imag'], values))

		if self.data is not None:
			self.data = self.data[self.perm]

	def num_batches(self):
		return int(math.floor(self.num_examples / self.batch_size))

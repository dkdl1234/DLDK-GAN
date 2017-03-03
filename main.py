'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib import layers, losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave
from nova_data_set import nova_set

from progressbar import ETA, Bar, Percentage, ProgressBar

from gan import GAN

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("updates_per_epoch", 100, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")   
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "gan", "Model type")
flags.DEFINE_string("load_opt", "ref", "Transmitance or Reflactance")

FLAGS = flags.FLAGS

if __name__ == "__main__":
	data_directory = FLAGS.working_directory
	if not os.path.exists(data_directory):
		os.makedirs(data_directory)

	#TODO : replace mnist data set and replace with NOVA dataset
	#Done
	#TODO : check integrity of the data set with model
	print('Creating NOVA set...')
	novaSet = nova_set(batch_size = FLAGS.batch_size, dir_path = data_directory)
	opt = FLAGS.load_opt
	print('Loading {} data...'.format(opt))
    	novaSet.load(opt)
	novaSet.load('data')

	#TODO : remove VAE option, set model to GAN
	#Done
	print('Creating GAN model...')
	model = GAN(FLAGS.learning_rate, FLAGS.batch_size ,log_dir='./logs/', model_dir='./model/')
    	num_batches = novaSet.num_batches()

	print('Start Training...')
	for epoch in range(FLAGS.max_epoch):
		training_loss = 0.0
		pbar = ProgressBar()
		novaSet.permutate()
		for i in pbar(range(num_batches)):
			#fetch the batch of data to the model
			ref_real, _  = novaSet.next_batch('ref')
	    		noise = np.random.normal(0.0, 0.02, size=[ref_real.shape[0], 8]).astype(np.float32)

			#feed the btach of data to the model
			loss_value = model.update_params(ref_real, noise)
			training_loss += loss_value

			training_loss = training_loss / \
				(FLAGS.updates_per_epoch * FLAGS.batch_size)

		if (epoch % 10) == 0 :
			print("Loss %f" % (training_loss))
			model.save_model(epoch)

		#novaSet.batch_counter = 0
		#data, _ = novaSet.next_batch('data')
		#model.generate_and_save_images(data, FLAGS.working_directory)

	model.close_session()

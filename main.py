'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

import math
import os
from os.path import join
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
flags.DEFINE_string("data_directory", "", "") 
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "gan", "Model type")
flags.DEFINE_string("load_opt", "all", "Transmitance or Reflactance")

FLAGS = flags.FLAGS

if __name__ == "__main__":
	if not os.path.exists(FLAGS.working_directory):
		os.makedirs(FLAGS.working_directory)

	print('Creating NOVA set...')
	novaSet = nova_set(batch_size = FLAGS.batch_size, dir_path = FLAGS.data_directory)
	opt = FLAGS.load_opt
	print('Loading {} data...'.format(opt))
    	novaSet.load(opt)

	if opt != 'all':
		novaSet.load('data')
	
	#set the running configuration
	rconfig = None
	if opt == 'all':
		rconfig = ['ref', 'tra']
	else:
		rconfig = [opt]

	for config in rconfig:
		for config_type in ['real', 'imag']:
			print('Creating GAN model for {} {}...'.format(config, config_type))
			model = GAN(	FLAGS.learning_rate, \
					FLAGS.batch_size ,\
					log_dir	  = join(FLAGS.working_directory, 'logs/') + join(config,config_type),\
					model_dir = join(FLAGS.working_directory, 'model/')+ join(config,config_type))

    			num_batches = novaSet.num_batches()

			print('Start training {} {} part...'.format(config, config_type))
			for epoch in range(FLAGS.max_epoch):
				training_loss = 0.0
				pbar = ProgressBar()
				novaSet.permutate()
				for i in pbar(range(num_batches)):
					#fetch the batch of data to the model
					data  = novaSet.next_batch(which_config = [config, 'data'], which_type = config_type)

					#feed the batch of data to the model
					loss_value = model.update_params(data[0], data[1])
					training_loss += loss_value

					training_loss = training_loss / \
						(FLAGS.updates_per_epoch * FLAGS.batch_size)
					
					if (i % 10) == 0:
						model.write_summaries(sum_epoch = i + (num_batches * epoch))
					del data

				if (epoch % 10) == 0 :
					print("Epoch %d :Loss %f" % (epoch, training_loss))
					model.save_model(epoch)

			print('Done optimizing {} {} part!'.format(config, config_type))
			print('Closeing session...')
			model.close()
			del model



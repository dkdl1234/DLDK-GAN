

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
from img_manip import merge_image_summaries

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
flags.DEFINE_string("load_opt", "all", "Transmitance or Reflactance")
flags.DEFINE_string("load_type", "all", "Real of Imaginary type")
flags.DEFINE_boolean("save_imgs", False, "") #set to false because entire image savings (100'000 images) is THE performance drawback
flags.DEFINE_boolean("save_train", True, "Save training images/testing images")
flags.DEFINE_integer("snum_imgs",5, "Number of images to save each end of training epoch")
flags.DEFINE_boolean("merge_imgs", False, "Merge flags, if true - merge the picture, else don't")

FLAGS = flags.FLAGS

if __name__ == "__main__":
	if not os.path.exists(FLAGS.working_directory):
		os.makedirs(FLAGS.working_directory)

	#create the nova data set
	print('Creating NOVA set...')
	novaSet = nova_set(batch_size = FLAGS.batch_size, dir_path = FLAGS.data_directory)

	#load the nove set data	
	opt 	 = FLAGS.load_opt
	opt_type = FLAGS.load_type
	print('Loading {} data...'.format(opt))
    	novaSet.load(opt)

	if opt != 'all':
		novaSet.load('data')

	#split the data into training and testing data
	novaSet.split_data(ratio=0.8)
	

	#set the running configuration
	rconfig = None
	if opt == 'all':
		rconfig = ['ref', 'tra']
	else:
		rconfig = [opt]

	#set the type of the matrix data (real or imaginary)
	tconfig = None
	if opt_type == 'all':
		tconfig = ['real', 'imag']
	else:
		tconfig = [opt_type]

	
	#get image indices to be saved as PDF image summaries later
	if FLAGS.save_train:
		image_indices = np.random.permutation(novaSet.num_train_examples)[:FLAGS.snum_imgs]
	else:
		image_indices = np.random.permutation(novaSet.num_examples - novaSet.num_train_examples)[:FLAGS.snum_imgs]

	#main computational efforts
	for config in rconfig:
		for config_type in tconfig:
			print('Creating GAN model for {} {}...'.format(config, config_type))
			model = GAN(	FLAGS.learning_rate, \
					log_dir	  = join(FLAGS.working_directory, "") + join(config,config_type + "/logs"),\
					model_dir = join(FLAGS.working_directory, "") + join(config,config_type + "/model"))

			print('Number of trainable variables of the model: {}'.format(model.num_vars()))
    			num_batches = novaSet.num_batches()
			

			#save real images at the beginning, before the heavy lifting begins
			if FLAGS.save_imgs:
				#save fake images
				images = novaSet.get_data(which_config = ['data'], which_type = None, indices = image_indices)
				
				image_dir = join(FLAGS.working_directory, "") + join(config,config_type + "/imgs")
				
				#save the real indexed images
				novaSet.save_images(	which_config=config,\
							which_type=config_type,\
							which_indices=image_indices,\
							save_dir = join(image_dir, "real")\
							)


			print('Start training {} {} part...'.format(config, config_type))
			for epoch in range(FLAGS.max_epoch):
				training_loss = 0.0
				pbar = ProgressBar()
				novaSet.permutate()
				for i in pbar(range(num_batches)):
					#fetch the batch of data to the model
					data  = novaSet.next_batch(which_config = [config, 'data'], which_type = config_type)
					real_images, nova_params = data[0], data[1]
					del data

					#feed the batch of data to the model
					loss_value = model.update_params(real_images, nova_params)
					training_loss += loss_value

					training_loss = training_loss / \
						(FLAGS.updates_per_epoch * FLAGS.batch_size)
					
					#every 10th batch, write summaries to disk
					if (i % 10) == 0:
						model.write_summaries(sum_epoch = i + (num_batches * epoch))
						model.calculate_accuracy(nova_params, epoch = i + (num_batches * epoch))
 
					del real_images, nova_params

				
				#calculate testing loss and accuracy
				inputs = novaSet.test_info(which_config = [config, 'data'], which_type = config_type)
				gen_tst_loss, gen_tst_accur, disc_tst_loss, disc_tst_accur = model.test(imgs=inputs[0], data=inputs[1], epoch=epoch)
				print("Epoch %d : Gen Test Loss %f,  Gen Test Accur %f, Disc Test Loss %f, Disc Test Accur %f, Gen Train Loss %f"\
					 % (epoch, gen_tst_loss, gen_tst_accur, disc_tst_loss, disc_tst_accur, training_loss))
				del inputs

				#every 10 epochs, write the models' parameters to the disk an save images
				if (epoch % 10) == 0 :
					model.save_model(epoch)

					#save images
					print("Saving Image Summaries")
					images = novaSet.next_batch(which_config = [config, 'data'], which_type = config_type, batch_size=5)
					model.save_image_summary(r_inputs=images[0], f_inputs=images[1], sum_epoch=epoch)
					del images

					if FLAGS.save_imgs:
						
						images = novaSet.get_data(which_config = ['data'],\
									which_type = None,\
									indices = image_indices,\
									is_train=FLAGS.save_train)
						print("Generating and saving images for epoch {}...".format(epoch))
						image_dir = join(FLAGS.working_directory, "") + join(config,config_type + "/imgs")
						model.generate_and_save_images(images[0], image_indices, join(image_dir, "fake"), epoch)
					
			print("Saving final model...")
			model.save_model(FLAGS.max_epoch + 1)
			
			print("Generating and saving final image summary")			
			images = novaSet.next_batch(which_config = [config, 'data'], which_type = config_type, batch_size=5)
			model.save_image_summary(r_inputs=images[0], f_inputs=images[1], sum_epoch=epoch)
			del images
			

			if FLAGS.save_imgs:
				print("Generating and saving images for final model")
				#save fake images
				images = novaSet.get_data(which_config = ['data'], which_type = None, indices = image_indices)
				
				image_dir = join(FLAGS.working_directory, "") + join(config,config_type + "/imgs")
				model.generate_and_save_images(	images[0], \
								image_indices, \
								join(image_dir, "fake"), \
								'final')
				
				if FLAGS.merge_imgs:
					#generate image summary
					merge_image_summaries(	real_dir=join(image_dir, "real"), \
								fake_dir=join(image_dir, "fake"), \
								targ_dir=image_dir)
							

			#close the model and delete it
			print('Done optimizing {} {} part!'.format(config, config_type))
			print('Closeing session...')
			model.close()
			del model



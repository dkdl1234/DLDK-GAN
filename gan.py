'''TensorFlow implementation of http://arxiv.org/pdf/1511.06434.pdf'''

from __future__ import absolute_import, division, print_function

import math

import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf
import os

from utils import discriminator, decoder
from generator import Generator

def concat_elu(inputs):
    return tf.nn.elu(tf.concat(3, [-inputs, inputs]))

class GAN(Generator):

    def __init__(self, learning_rate, log_dir, model_dir):
	#TODO: change 'self.input_tensor' dimensionality from 28*28 to 30*30
	#DONE
	tf.set_random_seed(seed=1)
	self.graph = tf.Graph()
	with self.graph.as_default():
		with tf.name_scope('Input'):
        		self.input_tensor = tf.placeholder(tf.float32, [None, 30 * 30], name='NOVAImage') 
			self.nova_input   = tf.placeholder(tf.float32, [None, 8], name = 'Slice-Params')

        	with arg_scope([layers.conv2d, layers.conv2d_transpose],
				activation_fn=concat_elu,
                       		normalizer_fn=layers.batch_norm,
                       		normalizer_params={'scale': True}):
            		with tf.variable_scope("Model"):
                		D1 = discriminator(self.input_tensor)  # positive examples
                		D_params_num = len(tf.trainable_variables())
                		G = decoder(self.nova_input)
                		self.sampled_tensor = G

           		with tf.variable_scope("Model", reuse=True):
                		D2 = discriminator(G)  # generated examples

        	D_loss = self.__get_discrinator_loss(D1, D2)
        	G_loss = self.__get_generator_loss(D2)

        	params = tf.trainable_variables()
        	D_params = params[:D_params_num]
        	G_params = params[D_params_num:]
        	#    train_discrimator = optimizer.minimize(loss=D_loss, var_list=D_params)
        	# train_generator = optimizer.minimize(loss=G_loss, var_list=G_params)
        	global_step = tf.contrib.framework.get_or_create_global_step()
        	self.train_discriminator = layers.optimize_loss(D_loss, global_step, learning_rate / 10, 'Adam', variables=D_params, update_ops=[])
        	self.train_generator 	 = layers.optimize_loss(G_loss, global_step, learning_rate, 'Adam', variables=G_params, update_ops=[])
	
		#create a tensorflow session and initialize all the variables
        	self.sess = tf.Session(graph=self.graph)
        	self.sess.run(tf.global_variables_initializer())

		self.log_dir, self.model_dir = log_dir, model_dir
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)

		if not os.path.exists(model_dir):
			os.makedirs(model_dir)

			
		self.writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
	

    def __get_discrinator_loss(self, D1, D2):
        '''Loss for the discriminator network

        Args:
            D1: logits computed with a discriminator networks from real images
            D2: logits computed with a discriminator networks from generated images

        Returns:
            Cross entropy loss, positive samples have implicit labels 1, negative 0s
        '''
        return (losses.sigmoid_cross_entropy(D1, tf.ones(tf.shape(D1))) +
                losses.sigmoid_cross_entropy(D2, tf.zeros(tf.shape(D1))))

    def __get_generator_loss(self, D2):
        '''Loss for the genetor. Maximize probability of generating images that
        discrimator cannot differentiate.

        Returns:
            see the paper
        '''
        return losses.sigmoid_cross_entropy(D2, tf.ones(tf.shape(D2)))

    def update_params(self, disc_inputs, gen_inputs):
        d_loss_value = self.sess.run(self.train_discriminator, feed_dict={self.input_tensor: disc_inputs, self.nova_input : gen_inputs})
	
        g_loss_value = self.sess.run(self.train_generator  , feed_dict={self.nova_input : gen_inputs})

        return g_loss_value

	
    def operate(self, nova_inputs):
	'''
	NOTICE: this method set to be used only after training the generator!
	
	Args:
		nova_inputs : a numpy or tensorflow of the shape [?, 8], e.g. an unknown (user-defned) number of data set,
		each sample of the set has 8 features
	Returns:
		a condensed (flattened) image of the real/imaginary reflactance/transmitance matrix, of size [1, 900]
	'''
	if self.sess is None:
		self.sess = tf.Session(graph=self.graph)
		self.sess.run(tf.global_variable_initializer())

	return sess.run(self.samples_tensor, feed_dict={self.nova_input : nova_inputs})

    def close(self):
	'''
		Close the session that holds the graph
	'''
	self.sess.close()
	self.sess = None

    def save_model(self):
	
	
	model_file_path = os.path.join(self.model_dir, "model.cktp")
	if os.path.exists(model_file_path):
		os.remove(model_file_path)

	self.saver = tf.train.Saver()
	self.saver.save(self.sess, os.path.join(self.model_dir, "model.cktp"))
	self.saver = None

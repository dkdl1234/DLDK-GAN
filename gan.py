'''TensorFlow implementation of http://arxiv.org/pdf/1511.06434.pdf'''

from __future__ import absolute_import, division, print_function

import math

import numpy as np
from tensorflow.contrib import layers, losses, metrics, slim
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf
import os
import math

from utils import discriminator, decoder
from generator import Generator

def concat_elu(inputs):
    return tf.nn.elu(tf.concat(3, [-inputs, inputs]))

class GAN(Generator):

    def __init__(self, learning_rate, log_dir, model_dir):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('Input'):
                self.input_tensor = tf.placeholder(tf.float32, [None, 30 * 30], name='NOVAImage')
                self.nova_input   = tf.placeholder(tf.float32, [None, 8], name = 'Slice-Params')

            with arg_scope([layers.conv2d, layers.conv2d_transpose],\
			    activation_fn=concat_elu,\
                       	    normalizer_fn=layers.batch_norm,\
                       	    normalizer_params={'scale': True},\
			    weights_initializer=layers.xavier_initializer_conv2d()\
			    #weights_regularizer=tf.nn.l2_loss\
				):
                with tf.variable_scope("Model"):
                    D1 = discriminator(self.input_tensor)  # positive examples
                    D_params_num = len(tf.trainable_variables())
                    G = decoder(self.nova_input)
                    self.sampled_tensor = G
		    self.discriminator = D1

		    params = tf.trainable_variables()
		    self.G_params_clear = params[D_params_num:]

                with tf.variable_scope("Model", reuse=True):
                    D2 = discriminator(G)  # generated examples

            self.D_loss = self.__get_discrinator_loss(D1, D2)
            self.G_loss = self.__get_generator_loss(D2)
	    self.D_accur = self.__get_discriminator_accuracy(D1)
	    self.G_accur = self.__get_generator_accuracy(D1)

            self.params = tf.trainable_variables()
	    self.params_info = slim.model_analyzer.analyze_vars(self.params, print_info=False)
            self.D_params = self.params[:D_params_num]
            self.G_params = self.params[D_params_num:]
            
            global_step = tf.contrib.framework.get_or_create_global_step()
            self.train_discriminator = layers.optimize_loss(self.D_loss, global_step, learning_rate / 10, 'Adam', variables=self.D_params, update_ops=[])
            self.train_generator     = layers.optimize_loss(self.G_loss, global_step, learning_rate, 'Adam', variables=self.G_params, update_ops=[])

	    #generate summaries
	    self.__generate_summaries()

            #create the saver
            self.mod_saver = tf.train.Saver({v.op.name : v for v in self.params})
	    self.gen_saver = tf.train.Saver({v.op.name : v for v in self.G_params_clear})
            
	    #create a tensorflow session and initialize all the variables
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())

	    #create the directories for the dump files
            self.log_dir, self.model_dir = log_dir, model_dir
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            self.writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())



    def __generate_summaries(self, max_imag_batch=5):
	#Define discriminator and generator loss
	self.d_loss_sum = tf.summary.scalar("Disc_trn_Loss", self.D_loss)
	self.g_loss_sum = tf.summary.scalar("Gen_trn_Loss", self.G_loss)
	self.d_accur = tf.summary.scalar("Disc_trn_accur", self.D_accur)
	self.merged_op = tf.summary.merge([self.d_loss_sum, self.g_loss_sum, self.d_accur])
	
	#Define image summary ops
	###fake images:
	self.g_images = tf.summary.image("Generated_Images", tf.reshape(self.sampled_tensor, shape=[-1, 30, 30, 1]), max_outputs=max_imag_batch)
	###real images:
	self.r_images_tensor = tf.reshape(self.input_tensor, shape=[-1, 30, 30, 1])
	self.r_images = tf.summary.image("Real_Images", self.r_images_tensor, max_outputs=max_imag_batch)
	#merge the images
	self.merged_image_op = tf.summary.merge([self.g_images, self.r_images])

	#create the generator accuracy	
	self.g_accuracy = tf.summary.scalar("Gen_trn_accur", self.G_accur)

	#create the tst summaries
	self.d_tst_loss_sum 	= tf.summary.scalar("Disc_tst_Loss"	, self.D_loss)
	self.g_tst_loss_sum 	= tf.summary.scalar("Gen_tst_Loss"	, self.G_loss)
	self.d_tst_accur_sum 	= tf.summary.scalar("Disc_tst_accur"	, self.D_accur)
	self.g_tst_accur_sum 	= tf.summary.scalar("Gen_tst_accur"	, self.G_accur)



    def __get_discriminator_accuracy(self, D1):
	return metrics.accuracy(tf.cast(tf.round(tf.nn.sigmoid(D1)), dtype=tf.int32), tf.ones_like(D1, dtype=tf.int32))



    def __get_generator_accuracy(self, D1):
	'''Accuracy metrics for the fake images created by the generator
	   The goal of the generator is to make images that will make the discriminator think that they are real, 
	   hence all the labels are ones (real images)
	'''
	return metrics.accuracy(tf.cast(tf.round(tf.nn.sigmoid(D1)), dtype=tf.int32), tf.ones_like(D1, dtype=tf.int32))




    def __get_discrinator_loss(self, D1, D2):
        '''Loss for the discriminator network

        Args:
            D1: logits computed with a discriminator networks from real images
            D2: logits computed with a discriminator networks from generated images

        Returns:
            Cross entropy loss, positive samples have implicit labels 1, negative 0s
        '''
        return (losses.sigmoid_cross_entropy(D1, tf.ones_like(D1, dtype=D1.dtype)) +
                losses.sigmoid_cross_entropy(D2, tf.zeros_like(D1, dtype=D1.dtype)))



    def __get_generator_loss(self, D2):
        '''Loss for the generator. Maximize probability of generating images that
        discrimator cannot differentiate.

        Returns:
            see the paper
        '''
        return losses.sigmoid_cross_entropy(D2, tf.ones_like(D2, dtype=D2.dtype))



    def update_params(self, disc_inputs, gen_inputs):
        d_loss_value, g_loss_value, self.summary = self.sess.run([self.train_discriminator, self.train_generator, self.merged_op], \
									feed_dict={self.input_tensor: disc_inputs, self.nova_input : gen_inputs})

	#run the generator optimizer the second time to keep in pace with the discriminator and prevent early discriminator convergence
	g_loss_value += self.sess.run(self.train_generator, feed_dict={self.nova_input : gen_inputs})
	g_loss_value += self.sess.run(self.train_generator, feed_dict={self.nova_input : gen_inputs})
        return g_loss_value / 3



    def calculate_accuracy(self, gen_inputs, epoch):
	#create the images using the 8 parameter inputs
	g_images = self.operate(gen_inputs)
	#calculate the discrimination accuracy
	_, accur_summary = self.sess.run([self.discriminator, self.g_accuracy], {self.input_tensor : g_images})
	self.writer.add_summary(accur_summary, epoch)
	

    def operate(self, nova_inputs):
        '''
        NOTICE: this method set to be used only after training the generator!
        '''
        return self.sess.run(self.sampled_tensor, feed_dict={self.nova_input : nova_inputs})


    def close(self):
        '''
	    Close the session that holds the graph
        '''
        self.sess.close()
        self.sess = None



    def save_model(self, step):
        model_file_path = os.path.join(self.model_dir, "model.ckpt")
        generator_file_path = os.path.join(self.model_dir, "generator.ckpt")

        self.mod_saver.save(self.sess, model_file_path, global_step=step)
	self.gen_saver.save(self.sess, generator_file_path, global_step=step)


    def close_session(self):
        if self.sess is not None:
            self.sess.close()

        self.sess = None



    def trainable_params(self):
        return self.params


    def num_vars(self):
	return self.params_info[0]


    def write_summaries(self, sum_epoch):
	self.writer.add_summary(self.summary, sum_epoch)

    

    def save_image_summary(self, r_inputs, f_inputs, sum_epoch):
	assert r_inputs is not None
	assert f_inputs is not None
	_, _,img_summary = self.sess.run([self.sampled_tensor, self.r_images_tensor, self.merged_image_op], \
							{self.nova_input : f_inputs,\
							self.input_tensor : r_inputs})

	self.writer.add_summary(img_summary, sum_epoch)
	

    def test(self, data, imgs, epoch):
	batch_size = 100
	num_batches = int(math.floor(max(data.shape) / batch_size))
	gen_loss = disc_loss = disc_accur = gen_accur = 0
	for i in range(num_batches):
		begin = i
		end = i + batch_size
		#create the imgs and calculate the generator loss
		f_imgs, g_loss, g_loss_sum = self.sess.run(	[self.sampled_tensor, self.G_loss, self.g_tst_loss_sum],\
								feed_dict={self.nova_input : data[begin:end,:]})
		gen_loss += g_loss

		#calculate the generator accuracy
		f_discs, g_accur, g_accur_sum = self.sess.run(	[self.discriminator, self.G_accur, self.g_tst_accur_sum],\
								feed_dict={self.input_tensor : f_imgs})
		gen_accur += g_accur

		#calculate the discriminator loss, accuracy
		_, d_loss, d_accur, d_loss_sum, d_accur_sum = self.sess.run([	self.discriminator,\
										self.D_loss,\
										self.D_accur,\
										self.d_tst_loss_sum,\
										self.d_tst_accur_sum],\
										feed_dict={self.input_tensor : imgs[begin:end],\
										 self.nova_input   : data[begin:end]})
		
		disc_loss += d_loss
		disc_accur += d_accur


	gen_loss, disc_loss, disc_accur, gen_accur = 	gen_loss / num_batches,\
							disc_loss / num_batches,\
							disc_accur / num_batches,\
							gen_accur / num_batches

	

	
	sums = [d_loss_sum, d_accur_sum, g_loss_sum, g_accur_sum]
	#write the summaries to the file
	for summary in sums:
		self.writer.add_summary(summary, epoch)

	return gen_loss, gen_accur, disc_loss, disc_accur

##########################################################################
# Author: Safa Messaoud                                                        #
# E-Mail: messaou2@illinois.edu                                                #
# Instituation: University of Illinois at Urbana-Champaign                     #
# Course: ECE 544_na Fall 2017                                                 #
# Date: July 2017                                                              #
#                                                                              #
# Description: the denoising convolutional autoencoder model                   #
#                                                                              #
#                                                                              #
##########################################################################

import tensorflow as tf
import numpy as np
import utils


class DAE(object):
    """
    Denoising Convolutional Autoencoder
    """

    def __init__(self, config):
        """
        Basic setup.
        Args:
            config: Object containing configuration parameters.
        """

        # Model configuration.
        self.config = config

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.original_images = None

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.noisy_images = None

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.reconstructed_images = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # Global step Tensor.
        self.global_step = None

        # A boolean indicating whether the current mode is 'training'.
        self.phase_train = True

    def build_inputs(self):
        """ Input Placeholders.
        define place holders for feeding (1) noise-free images, (2) noisy images and (3) a boolean variable 
        indicating whether you are in the training or testing phase
        Outputs:
            self.original_images
            self.noisy_images
            self.phase_train
        """
        images = tf.placeholder(tf.float32, [
                                None, self.config.image_height * self.config.image_width], name='images')
        noisy_images = tf.placeholder(tf.float32, [
            None, self.config.image_height * self.config.image_width], name='noisy_images')
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.original_images = images
        self.noisy_images = noisy_images
        self.phase_train = phase_train

    def build_model(self):
        """Builds the model.
        # implements the denoising auto-encoder. Feel free to experiment with different architectures.
        Explore the effect of 1) deep networks (i.e., more layers), 2) interlayer batch normalization and
        3) dropout, 4) pooling layers, 5) convolution layers, 6) upsampling methods (upsampling vs deconvolution), 
        7) different optimization methods (e.g., stochastic gradient descent versus stochastic gradient descent
        with momentum versus RMSprop.  
        Do not forget to scale the final output between 0 and 1. 
        Inputs:
            self.noisy_images
            self.original_images
        Outputs:
            self.total_loss
            self.reconstructed_images 
        """

        self.reconstructed_images = x_reconstructed

        # Compute losses.
        self.total_loss = tf.sqrt(tf.reduce_mean(
            tf.square(x_reconstructed - self.original_images)))

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_model()
        self.setup_global_step()

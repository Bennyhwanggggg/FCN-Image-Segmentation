import os
import tensorflow as tf
import numpy as np
from PIL import Image
from models.model import *
from utils.utils import *

PATH = os.path.dirname(os.path.realpath(__file__))
training_data_directory = 'Training_data/data_png'
training_mask_data_directory = 'Training_data/mask_bin'
testing_data_directory = 'Test_data/data_png'
testing_mask_data_directory = 'Test_data/mask_bin'


class ObjectRemover:
    def __init__(self, learning_rate=1e-6, train_epochs=1000, batch_size=1, eval_step=50):
        """
        Initiate Object Remove that uses FCN (Fully Convolutional Network) to detect
        desired objects in an image that needs to be removed.

        :param learning_rate: learning rate to use
        :param train_epochs: number of training epochs
        :param batch_size: batch size to use
        :param eval_step: evaluation steps
        """
        self.original_height = 1280
        self.original_width = 720
        self.image_height = int(np.ceil(self.original_height/32) * 32)
        self.image_width = int(np.ceil(self.original_width/32) * 32)

        """
        Neural Network Setup
        """
        self.learning_rate = learning_rate
        self.training_epochs = train_epochs
        self.batch_size = batch_size
        self.eval_step = eval_step
        self.test_batch = 1

        """
        Directories Setup
        """
        self.training_directory = os.path.join(PATH, training_data_directory)
        self.training_mask_directory = os.path.join(PATH, training_mask_data_directory)

        self.test_data_directory = os.path.join(PATH, testing_data_directory)
        self.test_mask_directory = os.path.join(PATH, testing_mask_data_directory)

        """
        Training and test parameter setup
        """
        self.training_data, self.traning_mask_gt = self.create_training_data(num_class=2)
        self.test_data, self.test_mask_gt = self.create_test_data(num_class=2)
        self.batch_num = self.training_data.shape[0]//self.batch_size
        self.test_batch_num = self.test_data.shape[0]//self.test_batch

    def create_training_data(self, num_class=2):
        """

        :param num_class: number of classes to use
        :return: training and mask data
        """
        training_data, training_mask_gt = create_data(self.training_directory,
                                                      self.training_mask_directory,
                                                      self.image_height,
                                                      self.image_width,
                                                      num_class)
        return training_data, training_mask_gt

    def create_test_data(self, num_class=2):
        """

        :param num_class: number of classes to use
        :return: test and mask data
        """
        test_data, test_mask_gt = create_data(self.test_data_directory,
                                              self.test_mask_directory,
                                              self.image_height,
                                              self.image_width,
                                              num_class)
        return test_data, test_mask_gt

    def set_placeholder(self, shape=[None, None, None, 2], mask_name='mask_gt', training_name='training'):
        """

        :param shape: shape of place holder
        :param mask_name: mask place holder name
        :param training_name: training place holder name
        :return: mask and training placeholders
        """
        mask_placeholder = tf.placeholder(tf.float32, shape=shape, name=mask_name)
        training_placeholder = tf.placeholder(tf.bool, name=training_name)
        return mask_placeholder, training_placeholder





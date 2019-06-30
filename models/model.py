import tensorflow as tf
import os

PATH = os.path.dirname(os.path.realpath(__file__))

def convolution(tensor, num_filters, postfix_name,
                activation=tf.nn.relu, padding='same', reuse=False):
    """

    :param tensor: 4D Tensor of [batch_size, H, W, C]
    :param num_filters:  number of filters as a list form of [int, int]
    :param postfix_name: postfix name
    :param activation: activation function to use
    :param padding: padding to use
    :param reuse: reuse boolean
    :return: output of the convolution operations
    """
    network = tensor
    with tf.variable_scope('{}'.format(postfix_name), reuse=reuse):
        for i, F in enumerate(num_filters):
            network = tf.layers.conv2d(network, F, (3, 3), activation=None, padding=padding, name='conv_{}'.format(i+1))
            network = activation(network, name='activation_{}'.format(i+1))
    return network


def up_convolution(tensor, num_filters, postfix_name,
                   activation=tf.nn.relu, reuse=False):
    """

    :param tensor: 4D Tensor of [batch_size, H, W, C]
    :param num_filters: number of filters as a list form of [int, int]
    :param postfix_name: postfix name
    :param activation: activation function to use
    :param reuse: reuse boolean
    :return: output of the up convolution operations by 2 times
    """
    network = tensor
    with tf.variable_scope('{}'.format(postfix_name), reuse=reuse):
        network = tf.layers.conv2d_transpose(network,
                                             filters=num_filters,
                                             kernel_size=3,
                                             strides=(2, 2),
                                             padding='same',
                                             name='upsample_{}'.format(postfix_name))
        network = activation(network, name='activation')
    return network



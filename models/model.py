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
    with tf.variable_scope(postfix_name, reuse=reuse):
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
    with tf.variable_scope(postfix_name, reuse=reuse):
        network = tf.layers.conv2d_transpose(network,
                                             filters=num_filters,
                                             kernel_size=3,
                                             strides=(2, 2),
                                             padding='same',
                                             name='upsample_{}'.format(postfix_name))
        network = activation(network, name='activation')
    return network


def maxpooling(tensor, postfix_name, pool_size=(2, 2), strides=(2, 2), padding='same'):
    """

    :param tensor: 4D Tensor of [batch_size, H, W, C]
    :param postfix_name: postfix name
    :param pool_size: pool size to use in tuple form
    :param strides: number of strides in tuple form
    :param padding: padding to use
    :return: output of max pooling 2D network
    """
    with tf.variable_scope(postfix_name):
        network = tf.layers.max_pooling2d(inputs=tensor,
                                          pool_size=pool_size,
                                          strides=strides,
                                          padding=padding,
                                          name='pool')
        return network


def concatenation(a, b, postfix_name):
    """

    :param a: tensor A
    :param b: tensor B
    :param postfix_name: variable scope postfix name
    :return: concatenation of two tensor
    """
    with tf.variable_scope(postfix_name):
        network = tf.concat([a, b], axis=1, name='concat')
    return network


def mask_generator(sess, training, reuse):
    """

    :param sess: sess to load
    :param training: training data
    :param reuse: reuse boolean
    :return: image input, keep probability, final network
    """
    # encoding
    tf.saved_model.loader.load(sess=sess,
                               tags=['vgg16'],
                               export_dir=os.path.join(PATH, 'models/VGG-16_mod2FCN_ImageNet-Classification/'))
    graph = tf.get_default_graph()

    image_input = sess.graph.get_tensor_by_name('image_input:0')
    keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
    vgg_pool5 = sess.graph.get_tensor_by_name('pool5:0')
    vgg_pool4 = sess.graph.get_tensor_by_name('pool4:0')
    vgg_pool3 = sess.graph.get_tensor_by_name('pool3:0')
    vgg_pool2 = sess.graph.get_tensor_by_name('pool2:0')
    vgg_pool1 = sess.graph.get_tensor_by_name('pool1:0')

    # decoding
    with tf.variable_scope('unet', reuse=reuse):
        network_6 = up_convolution(vgg_pool5, 512, postfix_name='layer_6', activation=tf.nn.relu, reuse=reuse)
        network_6 = concatenation(network_6, vgg_pool4, postfix_name='layer_6')
        network_6 = convolution(network_6, [512, 512, 512], padding='same', postfix_name='layer_6',
                                activation=tf.nn.relu, reuse=reuse)

        network_7 = up_convolution(network_6, 256, postfix_name='layer_7', activation=tf.nn.relu, reuse=reuse)
        network_7 = concatenation(network_7, vgg_pool3, postfix_name='layer_7')
        network_7 = convolution(network_7, [256, 256, 256], padding='same', postfix_name='layer_7',
                                activation=tf.nn.relu, reuse=reuse)

        network_8 = up_convolution(network_7, 128, postfix_name='layer_8', activation=tf.nn.relu, reuse=reuse)
        network_8 = concatenation(network_8, vgg_pool2, postfix_name='layer_8')
        network_8 = convolution(network_8, [128, 128], padding='same', postfix_name='layer_6',
                                activation=tf.nn.relu, reuse=reuse)

        network_9 = up_convolution(network_8, 64, postfix_name='layer_9', activation=tf.nn.relu, reuse=reuse)
        network_9 = concatenation(network_9, vgg_pool1, postfix_name='layer_9')
        network_9 = convolution(network_9, [64, 64], padding='same', postfix_name='layer_6',
                                activation=tf.nn.relu, reuse=reuse)

        network_10 = up_convolution(network_9, 64, postfix_name='layer_10', activation=tf.nn.relu, reuse=reuse)
        network_10 = convolution(network_10, [64, 64, 2], padding='same', postfix_name='layer_6',
                                 activation=tf.nn.relu, reuse=reuse)

    return image_input, keep_prob, network_10

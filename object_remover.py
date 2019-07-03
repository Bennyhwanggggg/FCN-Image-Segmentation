import os

PATH = os.path.dirname(os.path.realpath(__file__))
training_data_directory = 'Training_data/data_png'
training_mask_data_directory = 'Training_data/mask_bin'
testing_data_directory = 'Test_data/data_png'
testing_mask_data_directory = 'Test_data/mask_bin'

from models.model import *
from utils.utils import *


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
        self.training_data, self.training_mask_gt = self.create_training_data(num_class=2)
        self.test_data, self.test_mask_gt = self.create_test_data(num_class=2)
        self.batch_num = self.training_data.shape[0]//self.batch_size
        self.test_batch_num = self.test_data.shape[0]//self.test_batch

        """
        Session
        """
        self.session_config = tf.ConfigProto()
        self.session = None

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

    def create_session(self):
        """Updates gpu option to allow growth

        :return: tensorflow session object
        """
        self.session_config.gpu_options.allow_growth = True
        session = tf.Session(config=self.session_config)
        return session

    def set_session(self, session):
        """

        :param session: tensorflow session object
        :return: None
        """
        self.session = session

    def generate_segmentation_mask(self, reuse=False):
        """Generate segmentation mask

        :param reuse: reuse boolean parameter
        :return: image input result, keep probability, mask result
        """
        image_input, keep_probability, mask = mask_generator(sess=self.session, training=self.training_data, reuse=reuse)
        return image_input, keep_probability, mask

    def mask_crop_bounding(self, mask, mask_gt, image, x=0, y=0):
        """

        :param mask: mask to crop to bounding box to
        :param mask_gt: mask_gt to use
        :param image: image to crop
        :param x: x cord to crop
        :param y: y cord to crop
        :return: cropped image
        """
        mask = tf.image.crop_to_bounding_box(mask, x, y, self.original_height, self.original_width)
        mask_gt_new = tf.image.crop_to_bounding_box(mask_gt, x, y, self.original_height, self.original_width)

        image_in_crop = tf.image.crop_to_bounding_box(image, x, y, self.original_height, self.original_width)

        return mask, mask_gt_new, image_in_crop

    def mark_loss(self, mask, mask_gt, num_class=3):
        """

        :param mask: mask to use
        :param mask_gt: mask_gt to use
        :param num_class: num class to use
        :return:
        """
        mask_logits = tf.reshape(mask, (-1, num_class))
        mask_label = tf.reshape(mask_gt, (-1, num_class))

        mask_acc = self.evaluation_metrics(mask_logits=mask_logits, mask_label=mask_label)

        mask_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mask_logits, labels=mask_label), name='mask_loss')
        softmax = tf.nn.softmax(mask_logits)
        predict = tf.reshape(tf.argmax(softmax, 1), (-1, self.original_height, self.original_width), name='final_pred')
        mask_optimised = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(mask_loss)
        return predict, mask_optimised, mask_acc

    def evaluation_metrics(self, mask_logits, mask_label):
        correct_prediciton = tf.equal(tf.argmax(tf.nn.softmax(mask_logits), 1), tf.argmax(mask_label, 1))
        mask_acc = tf.reduce_mean(tf.cast(correct_prediciton, tf.float32), name='accuracy')
        return mask_acc

    def run_session(self):
        if self.session is None:
            return
        init = tf.global_variables_initializer()
        self.session.run(init)

    def onehot_output(self, prediction):
        return one_hot_output(prediction)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(sess=self.session, save_path=path)





from PIL import Image
import numpy as np
import os


def image_padding(image, label, image_height, image_width):
    shape = image.shape
    assert image_height >= shape[0]
    assert image_width >= shape[1]

    offset_x, offset_y = 0, 0

    new_image = np.zeros([image_height, image_width, 3])
    new_image[offset_x: offset_x + shape[0], offset_y: offset_y + shape[1]] = image

    new_label = np.zeros([image_height, image_width])
    new_label[offset_x: offset_x + shape[0], offset_y: offset_y + shape[1]] = label

    return new_image, new_label


def one_hot_init_v2(dp, num_class):
    out = np.zeros((dp.size, num_class), dtype=np.uint8)
    out[np.arange(dp.size), dp.ravel().astype('uint8')] = 1
    out.shape = dp.shape + (num_class,)
    return out


def one_hot_output(dp, num_class=3):
    out = np.zeros((dp.size, num_class), dtype=np.uint8)
    out[np.arange(dp.size), dp.ravel()] = 255
    out.shape = dp.shape + (num_class,)
    return out
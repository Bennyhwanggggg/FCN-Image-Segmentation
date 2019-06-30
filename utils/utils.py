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


def crop(image, top, bottom, left, right):
    return image[top:bottom, left:right]


def create_data(data_directory, mask_directory, image_height, image_width, num_class):
    data, mast_gt = [], []
    for image_name in os.listdir(data_directory):
        if not image_name.endswith('.png'):
            continue
        data_path = os.path.join(data_directory, image_name)
        mask_path = os.path.join(mask_directory, image_name)

        data_ori = np.array(Image.open(data_path))
        label_ori = np.array(Image.open(mask_path))

        data_padding, label_padding = image_padding(data_ori, label_ori, image_height, image_width)

        data = data + [data_padding]
        mask_gt = mask_gt + [one_hot_init_v2(label_padding/255, num_class)]
    return np.array(data), np.array(mask_gt)


def binarizing(image, threshold):
    pixel_data = image.load()
    width, height = image.size
    for y in range(height):
        for x in range(width):
            pixel_data[x, y] = 0 if pixel_data[x, y] < threshold else 255
    return image


def to_binary(image_path, save_path):
    image = Image.open(image_path)
    image = image.convert("L")
    image = binarizing(image, 150)
    image.save(save_path)


def jpg2png(image_path, save_path):
    image = Image.open(image_path)
    image.save(save_path)


def rotate(image_path, degrees_to_rotate, save_path):
    image_object = Image.open(image_path)
    rotated_image = image_object.rotate(degrees_to_rotate)
    rotated_image.save(save_path)


def flip(image_path, flip_direction, save_path):
    flip_map = {
        'vertical': flip_vertically,
        'horizontal': flip_horizontally
    }

    flip_direction = flip_direction.lower()
    if flip_direction in flip_map.keys():
        flip_map[flip_direction](image_path, save_path)


def flip_vertically(image_path, save_path):
    image_object = Image.open(image_path)
    flipped_image = image_object.transpose(Image.FLIP_TOP_BOTTOM)
    flipped_image.save(save_path)


def flip_horizontally(image_path, save_path):
    image_object = Image.open(image_path)
    flipped_image = image_object.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_image.save(save_path)


def augment(directory_path):
    for image in os.listdir(directory_path):
        image_path = os.path.join(directory_path, image)
        rotate(image_path, 180, '{}_rotate.png'.format(image_path.strip('.png')))
        flip(image_path, 'vertical', '{}_flip_v.png'.format(image_path.strip('.png')))
        flip(image_path, 'horizontal', '{}_flip_h.png'.format(image_path.strip('.png')))

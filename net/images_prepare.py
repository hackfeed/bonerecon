import os
from zipfile import ZipFile

import numpy as np
from natsort import natsorted
from PIL import Image


def convert_one_channel(img):
    if len(img.shape) > 2:
        img = img[:, :, 0]
        return img

    return img


def pre_images(resize_shape, path, include_zip):
    if include_zip == True:
        ZipFile(path+'/dataset.zip').extractall(path)
        path = path+'/Images'
    dirs = natsorted(os.listdir(path))
    sizes = np.zeros([len(dirs), 2])
    images = img = Image.open(path+'/'+dirs[0])
    sizes[0, :] = images.size
    images = (images.resize((resize_shape), Image.ANTIALIAS))
    images = convert_one_channel(np.asarray(images))

    for i in range(1, len(dirs)):
        img = Image.open(path+'/'+dirs[i])
        sizes[i, :] = img.size
        img = img.resize((resize_shape), Image.ANTIALIAS)
        img = convert_one_channel(np.asarray(img))
        images = np.concatenate((images, img))
    images = np.reshape(images, (len(dirs), resize_shape[0], resize_shape[1], 1))

    return images, sizes

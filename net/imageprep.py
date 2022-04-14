import os

import numpy as np
from natsort import natsorted
from PIL import Image

from image import convert_one_channel


def pre_images(resize_shape, path):
    dirs = natsorted(os.listdir(path))
    images = img = Image.open(f'{path}/{dirs[0]}')
    images = convert_one_channel(np.asarray(images.resize((resize_shape), Image.ANTIALIAS)))

    for i in range(1, len(dirs)):
        img = Image.open(f'{path}/{dirs[i]}')
        img = img.resize((resize_shape), Image.ANTIALIAS)
        img = convert_one_channel(np.asarray(img))
        images = np.concatenate((images, img))

    images = np.reshape(images, (len(dirs), resize_shape[0], resize_shape[1], 1))

    return images

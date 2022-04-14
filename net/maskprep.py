import os

import numpy as np
from natsort import natsorted
from PIL import Image

from image import convert_one_channel


def pre_masks(path='masks'):
    dirs = natsorted(os.listdir(path))
    masks = img = Image.open(f'{path}/{dirs[0]}')
    masks = convert_one_channel(np.asarray(masks))

    for i in range(1, len(dirs)):
        img = Image.open(f'{path}/{dirs[i]}')
        img = convert_one_channel(np.asarray(img))
        masks = np.concatenate((masks, img))

    masks = np.reshape(masks, (len(dirs), 512, 512, 1))

    return masks

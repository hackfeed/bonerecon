import os
import sys
from zipfile import ZipFile

import numpy as np
from natsort import natsorted
from PIL import Image

script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
default_path = script_dir+'/original_masks'


def convert_one_channel(img):
    if len(img.shape) > 2:
        img = img[:, :, 0]
        return img
    else:
        return img


def pre_masks(resize_shape=(512, 512), path=default_path):
    ZipFile(path+"/original_masks.zip").extractall(path+'/masks')
    path = path+'/masks'
    dirs = natsorted(os.listdir(path))
    masks = img = Image.open(path+'/'+dirs[0])
    masks = (masks.resize((resize_shape), Image.ANTIALIAS))
    masks = convert_one_channel(np.asarray(masks))
    for i in range(1, len(dirs)):
        img = Image.open(path+'/'+dirs[i])
        img = img.resize((resize_shape), Image.ANTIALIAS)
        img = convert_one_channel(np.asarray(img))
        masks = np.concatenate((masks, img))
    masks = np.reshape(masks, (len(dirs), resize_shape[0], resize_shape[1], 1))
    return masks


default_path = script_dir+'/custom_masks'


# CustomMasks 512x512
def pre_splitted_masks(path=default_path):
    ZipFile(path+"/masks.zip").extractall(path+'/masks')
    path = path+'/masks'
    dirs = natsorted(os.listdir(path))
    masks = img = Image.open(path+'/'+dirs[0])
    masks = convert_one_channel(np.asarray(masks))
    for i in range(1, len(dirs)):
        img = Image.open(path+'/'+dirs[i])
        img = convert_one_channel(np.asarray(img))
        masks = np.concatenate((masks, img))
    masks = np.reshape(masks, (len(dirs), 512, 512, 1))
    return masks

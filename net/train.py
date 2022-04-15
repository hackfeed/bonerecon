import os

import albumentations as A
import cv2
import numpy as np
from tensorflow.keras.models import save_model

from dataset import download_dataset, prepare_dataset, prepare_masks
from imageprep import pre_images
from maskprep import pre_masks
from model import UNet


def main():
    if not os.path.exists('data/dataset.zip'):
        download_dataset('data')

    prepare_dataset('data/dataset.zip', 'data')
    prepare_masks('masks/teeth.zip', 'masks/teeth')
    prepare_masks('masks/mandibles.zip', 'masks/mandibles')

    X = pre_images((512, 512), 'data/Images')
    Y = pre_masks('masks/teeth')

    X = np.float32(X/255)
    Y = np.float32(Y/255)

    x_train = X
    y_train = Y

    aug = A.Compose([
        A.OneOf([A.RandomCrop(width=512, height=512),
                A.PadIfNeeded(min_height=512, min_width=512, p=0.5)], p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.Compose([A.RandomScale(scale_limit=(-0.15, 0.15), p=1, interpolation=1),
                   A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT),
                   A.Resize(512, 512, cv2.INTER_NEAREST), ], p=0.5),
        A.ShiftScaleRotate(shift_limit=0.325, scale_limit=0.15, rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, p=1),
        A.Rotate(15, p=0.5),
        A.Blur(blur_limit=1, p=0.5),
        A.Downscale(scale_min=0.15, scale_max=0.25,  always_apply=False, p=0.5),
        A.GaussNoise(var_limit=(0.05, 0.1), mean=0, per_channel=True, always_apply=False, p=0.5),
        A.HorizontalFlip(p=0.25),
    ])

    x_train1 = np.copy(x_train)
    y_train1 = np.copy(y_train)

    count = 0
    while(count < 4):
        x_aug2 = np.copy(x_train1)
        y_aug2 = np.copy(y_train1)

        for i in range(len(x_train1)):
            augmented = aug(image=x_train1[i, :, :, :], mask=y_train1[i, :, :, :])
            x_aug2[i, :, :, :] = augmented['image']
            y_aug2[i, :, :, :] = augmented['mask']
        x_train = np.concatenate((x_train, x_aug2))
        y_train = np.concatenate((y_train, y_aug2))

        if count == 9:
            break

        count += 1

    mem_to_free = [x_aug2, X, Y, y_aug2, y_train1, x_train1, augmented]
    for mem in mem_to_free:
        del mem

    model = UNet(input_shape=(512, 512, 1), last_activation='sigmoid')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=16, epochs=100, verbose=1)

    save_model(model, 'trained/bonerecon.h5')


if __name__ == '__main__':
    main()

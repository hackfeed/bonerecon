import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from CCA_Analysis import *
from download_dataset import *
from images_prepare import *
from masks_prepare import *
from model import *

if not os.path.exists('data/dataset.zip'):
    download_dataset('data/')

X, X_sizes = pre_images((512, 512), 'data', True)

Y = pre_splitted_masks(path='custom_masks')

X = np.float32(X/255)
Y = np.float32(Y/255)

x_train = X[:105, :, :, :]
y_train = Y[:105, :, :, :]
x_test = X[105:, :, :, :]
y_test = Y[105:, :, :, :]

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

del x_aug2
del X
del Y
del y_aug2
del y_train1
del x_train1
del augmented


model = UNET(input_shape=(512, 512, 1), last_activation='sigmoid')
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=8, epochs=1, verbose=1)

predict_img = model.predict(x_test)
predict = predict_img[1, :, :, 0]

predict_img1 = (predict_img > 0.25)*1
y_test1 = (y_test > 0.25)*1

f1_score(predict_img1.flatten(), y_test1.flatten(), average='micro')

plt.figure(figsize=(20, 10))
plt.title('Predict Mask', fontsize=40)
plt.imshow(predict)
plt.imsave('predict.png', predict)

img = cv2.imread('data/Images/107.png')

predicted = cv2.imread('data/predict.png')

predicted = cv2.resize(predicted, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

cca_result, teeth_count = CCA_Analysis(img, predicted, 3, 2)
cv2.imshow(cca_result)

print(teeth_count, 'Teeth Count')

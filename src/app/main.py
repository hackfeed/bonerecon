import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cca import analyze
from image import convert_one_channel, convert_rgb
from PIL import Image
from tensorflow.keras.models import load_model


def main(model, image, output_dir):
    model = load_model(model)
    img = Image.open(image)
    img = np.asarray(img)

    img_cv = convert_one_channel(img)
    img_cv = cv2.resize(img_cv, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    img_cv = np.float32(img_cv/255)
    img_cv = np.reshape(img_cv, (1, 512, 512, 1))

    predicted = model.predict(img_cv)[0]
    predicted = cv2.resize(
        predicted, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    plt.imsave(f'{output_dir}/predicted.png', predicted)

    thresh = np.uint8(predicted*255)
    thresh = cv2.threshold(thresh, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    kernel = (np.ones((5, 5), dtype=np.float32))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    output = cv2.drawContours(convert_rgb(img), cnts, -1, (255, 0, 0), 3)
    plt.imsave(f'{output_dir}/segmented.png', output)

    img = cv2.imread(image)
    predicted = cv2.imread(f'{output_dir}/predicted.png')
    predicted = cv2.resize(
        predicted, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    cca, teeth_count = analyze(img, predicted, 3, 2, True)
    plt.imsave(f'{output_dir}/segmented_cca.png', cca)
    print(f'Segmented teeth count is {teeth_count}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='trained/bonerecon.h5')
    parser.add_argument('-i', '--image', default='data/file.jpeg')
    parser.add_argument('-o', '--output', default='predicted')

    args = parser.parse_args()

    main(args.model, args.image, args.output)

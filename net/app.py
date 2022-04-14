import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from cca import analyze
from image import convert_one_channel, convert_rgb


def main():
    model = load_model("trained/bonerecon.h5")
    img = Image.open("data/file.jpeg")
    img = np.asarray(img)

    img_cv = convert_one_channel(img)
    img_cv = cv2.resize(img_cv, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    img_cv = np.float32(img_cv/255)
    img_cv = np.reshape(img_cv, (1, 512, 512, 1))

    predicted = model.predict(img_cv)[0]
    predicted = cv2.resize(
        predicted, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    plt.imsave('predicted.png', predicted)

    thresh = np.uint8(predicted*255)
    thresh = cv2.threshold(thresh, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    kernel = (np.ones((5, 5), dtype=np.float32))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    output = cv2.drawContours(convert_rgb(img), cnts, -1, (255, 0, 0), 3)
    plt.imsave('res.png', output)

    img = cv2.imread('data/file.jpeg')
    predicted = cv2.imread('predicted.png')
    predicted = cv2.resize(
        predicted, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    cca, teeth_count = analyze(img, predicted, 3, 2)
    plt.imsave('res_cca.png', cca)
    print(f'Segmented teeth count is {teeth_count}')


if __name__ == '__main__':
    main()

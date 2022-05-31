from PIL import Image
import numpy as np


def convert_one_channel(img):
    if len(img.shape) > 2:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def main():
    oimg = Image.open('/Users/kononenko/Diploma/bonerecon/docs/diploma/body/img/blue.jpeg')
    oimg.resize((512, 512)).save('resized.png')
    oimg.resize((512, 512)).convert('L').save('converted.png')


if __name__ == '__main__':
    main()

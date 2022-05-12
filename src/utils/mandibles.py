import os

from natsort import natsorted
from PIL import Image


def main():
    images = natsorted(os.listdir('data/Segmentation1'))

    for img in images:
        oimg = Image.open(f'data/Segmentation1/{img}')
        grayed = oimg.convert('L')
        bw = grayed.point(lambda x: 0 if x < 40 else 255, '1')
        bw.save(f'masks/mandibles/{img}')

    images = natsorted(os.listdir('masks/mandibles'))

    for img in images:
        oimg = Image.open(f'masks/mandibles/{img}')
        oimg = oimg.resize((512, 512))
        oimg.save(f'masks/mands/{img}')


if __name__ == '__main__':
    main()

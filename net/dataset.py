import os
from io import BytesIO
from zipfile import ZipFile

import requests

DATASET_LINK = 'https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/hxt48yk462-1.zip'


def download_dataset(path):
    print(f'Downloading dataset to {path}...')
    r = requests.get(DATASET_LINK)
    z = ZipFile(BytesIO(r.content))
    z.extractall(path)
    os.rename(f'{path}/{z.namelist()[0]}', f'{path}/dataset.zip')

    print(f'Downloaded to {path}/dataset.zip')


def prepare_dataset(path='data/dataset.zip'):
    print(f'Extracting {path}...')
    extract_to = path.split('/')[0]
    ZipFile(path).extractall(extract_to)
    print(f'Extracted to {extract_to}')


def prepare_masks(path='masks/masks.zip'):
    print(f'Extracting {path}...')
    extract_to = path.split('/')[0]
    extract_to = f'{extract_to}/{extract_to}'

    if not os.path.exists(extract_to):
        os.mkdir(extract_to)

    ZipFile(path).extractall(extract_to)
    print(f'Extracted to {extract_to}')

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


def prepare_dataset(dataset_path='data/dataset.zip', masks_path='masks/masks.zip'):
    print(f'Extracting {dataset_path}...')
    extract_to = dataset_path.split('/')[0]
    ZipFile(dataset_path).extractall(extract_to)
    print(f'Extracted to {extract_to}')

    print(f'Extracting {masks_path}...')
    extract_to = masks_path.split('/')[0]
    ZipFile(masks_path).extractall(extract_to)
    print(f'Extracted to {extract_to}')

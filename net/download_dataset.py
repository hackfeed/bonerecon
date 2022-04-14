import os
from io import BytesIO
from zipfile import ZipFile

import requests


def download_dataset(save_path):
    r = requests.get(
        "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/hxt48yk462-1.zip")
    print("Downloading...")
    z = ZipFile(BytesIO(r.content))
    z.extractall(save_path)
    os.rename(save_path+z.namelist()[0], save_path+"dataset.zip")
    print("Completed...")

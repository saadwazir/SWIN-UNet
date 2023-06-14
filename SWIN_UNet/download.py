import numpy as np
from glob import glob

from swin_unet import swin_unet_2d_base

import sys
filepath = './dataset/oxford_iiit/'

from pathlib import Path

Path(filepath).mkdir(parents=True,exist_ok=True)


    # downloading and executing data files
import tarfile
import urllib.request

filename_image = filepath+'images.tar.gz'
filename_target = filepath+'annotations.tar.gz'

urllib.request.urlretrieve('http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz', filename_image);
urllib.request.urlretrieve('https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz', filename_target);

with tarfile.open(filename_image, "r:gz") as tar_io:
    tar_io.extractall(path=filepath)
with tarfile.open(filename_target, "r:gz") as tar_io:
    tar_io.extractall(path=filepath)

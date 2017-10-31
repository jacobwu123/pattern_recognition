"""Input and output helpers to load in data.
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.
    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.
    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,28,28)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """

    file= open(data_txt_file, 'r')
    data = {}
    file_size = sum(1 for _ in file)
    images = np.ndarray(shape = (file_size,28,28))
    labels = np.ndarray(shape = (file_size,))
    count = 0
    for line in file.readlines():
        image_name = line.strip().split('\t')[0]    
        images[count] = io.imread(image_data_path+'/'+image_name,True)
        labels[count] = int(line.strip().split('\t')[1])
        count += 1
    file.close()
    data['image'] = images
    data['label'] = labels

    return data


def write_dataset(data_txt_file, data):
    """Write python dictionary data into csv format for kaggle.
    Args:
        data_txt_file(str): path to the data txt file.
        data(dict): A Python dictionary with keys 'image' and 'label',
          (see descriptions above).
    """
    pass

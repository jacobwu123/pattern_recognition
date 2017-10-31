"""
Implements feature extraction and other data processing helpers.
"""

import numpy as np
import skimage
from skimage import filters
from numpy import fft


def preprocess_data(data, process_method='default'):
    """
    Args:
        data(dict): Python dict loaded using io_tools.
        process_method(str): processing methods needs to support
          ['raw', 'default'].
        if process_method is 'raw'
          1. Convert the images to range of [0, 1]
          2. Remove mean.
          3. Flatten images, data['image'] is converted to dimension (N, 28*28)
        if process_method is 'default':
          1. Convert images to range [0,1]
          2. Apply laplacian filter with window size of 11x11. (use skimage)
          3. Remove mean.
          3. Flatten images, data['image'] is converted to dimension (N, 28*28)

    Returns:
        data(dict): Apply the described processing based on the process_method
        str to data['image'], then return data.
    """
    if process_method == 'default':
        #convert the images to range of [0,1]
        file_size  = len(data['image'])
        for x in range(file_size):
            data['image'][x] = skimage.img_as_float(data['image'][x])
        
        data = remove_data_mean(data)
        flatten_images = np.zeros(shape=(len(data['image']),28*28))
        for x in range(file_size):
            flatten_images[x] = data['image'][x].flatten()
        data['image'] = flatten_images

    elif process_method == 'raw':
        file_size  = len(data['image'])
        for x in range(file_size):
            data['image'][x] = skimage.img_as_float(data['image'][x])
            filters.laplace(data['image'][x],11)
        data = remove_data_mean(data)
        flatten_images = np.zeros(shape=(len(data['image']),28*28))
        for x in range(file_size):
            flatten_images[x] = data['image'][x].flatten()
        data['image'] = flatten_images
            
    elif process_method == 'custom':
        pass

    return data


def compute_image_mean(data):
    """ Computes mean image.
    Args:
        data(dict): Python dict loaded using io_tools.
    Returns:
        image_mean(numpy.ndarray): Avaerage across the example dimension.
    """

    image_mean = np.mean(data['image'],axis = 0)

    return image_mean


def remove_data_mean(data):
    """
    Args:
        data(dict): Python dict loaded using io_tools.
    Returns:
        data(dict): Remove mean from data['image'] and return data.
    """
    for x in range(len(data['image'])):
        data['image'][x] -= compute_image_mean(data)

    return data

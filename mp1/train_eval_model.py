"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression


def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    batch_epoch_num = data['label'].shape[0] // batch_size
    np.random.shuffle(data['image'])
    count = 0
    for y in range(num_steps):    
        #load data
        #update model
        update_step(data['image'][count*batch_size:(count+1)*batch_size,:],
                    data['label'][count*batch_size:(count+1)*batch_size],
                    model, learning_rate)
        count += 1
        if count >= batch_epoch_num:
        #shuffle:
            np.random.shuffle(data['image'])
            count = 0

    return model


def update_step(image_batch, label_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).
    Args:
        image_batch(numpy.ndarray): input data of dimension (N, ndims).
        label_batch(numpy.ndarray): label data of dimension (N,).
        model(LinearModel): Initialized linear model.
    """
    ##update model
    f = model.forward(image_batch)
    gradient = model.backward(f,label_batch)
    model.w = model.w - learning_rate*gradient

def eval_model(data, model):
    """Performs evaluation on a dataset.
    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    f = model.forward(data['image'])
    loss = model.loss(f,data['label'])
    acc = 0
    for x in range(f.shape[0]):
        if data['label'][x] == f[x]:
            acc += 1
        
    acc /= (1.0*f.shape[0])
    return loss, acc

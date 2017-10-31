"""
Implements lienar regression.
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class LinearRegression(LinearModel):
    """Implements a linear regression mode model"""

    def backward(self, f, y):
        """Performs the backward operation.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
            y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
            (numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,).
        """

        gradient = np.dot(self.x.T ,f) - np.dot(self.x.T , y)
        return gradient

    def loss(self, f, y):
        """The average loss across batch examples.
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
            y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
            (float): average square loss.
        """
        err_sum = 0
        for x in range(f.shape[0]):
            err_sum += ((1-y[x]*f[x])**2)
    
        return 0.5*err_sum

    def predict(self, f):
        """Converts score into predictions in {-1, 1}.
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,).
        """
        y_predict = np.ndarray(shape=(f.shape))
        for x in range(f.shape[0]):
            if f[x] >= 0:
                y_predict[x] = 1
            else:
                y_predict[x] = -1


        return y_predict

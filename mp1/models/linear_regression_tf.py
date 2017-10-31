"""Linear regression model implemented in TensorFlow.
"""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from models.linear_model_tf import LinearModelTf


class LinearRegressionTf(LinearModelTf):
    def loss(self, f, y):
        """The average loss across batch examples.
        Computes the average square error.
        Args:
            f: Tensor containing the output of the forward operation.
            y(tf.placeholder): Tensor containing the ground truth label.
        Returns:
            (1): Returns the loss function tensor.
        """
        return tf.reduce_sum(tf.square(y-f))

    def predict(self, f):
        """Converts score into predictions in {-1, 1}
        Args:
            f: Tensor containing theoutput of the forward operation.
        Returns:
            (1): Converted predictions, tensor of the same dimension as f.
        """
 
        return tf.sign(f)

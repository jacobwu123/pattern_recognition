"""Implements the k-means algorithm.
"""

import numpy as np
import scipy
from scipy import stats


class KMeans(object):
    def __init__(self, n_dims, n_components=10, max_iter=100):
        """Initialize a KMeans GMM model
        Args:
            n_dims(int): The dimension of the feature.
            n_components(int): The number of cluster in the model.
            max_iter(int): The number of iteration to run EM.
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter

        # Randomly initialize model parameters using Gaussian distribution of 0
        # mean and unit variance.
        mean = 0.
        var = 1.
        self._mu = np.random.normal(mean,var,(self._n_components,self._n_dims))  # np.array of size (n_components, n_dims)

    def fit(self, x):
        """Runs EM step for max_iter number of times.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
            ndims).
        """
        for i in range(self._max_iter):
            r_ik = self._e_step(x)
            self._m_step(x,r_ik)

    def _e_step(self, x):
        """Update cluster assignment.

        Computes the posterior probability of p(z|x), z is the latent cluster
        variable.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            r_ik(numpy.ndarray): Array containing the cluster assignment of
                each example, dimension (N,).
        """
        r_ik = self.get_posterior(x)
        return r_ik

    def _m_step(self, x, r_ik):
        """Update cluster mean.

        Updates self_mu according to the cluster assignment.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            r_ik(numpy.ndarray): Array containing the cluster assignment of
                each example, dimension (N,).
        """
        np.seterr(divide='ignore', invalid='ignore')
        N, = r_ik.shape
        for k in range(self._n_components):
            count = 0
            accumlator = np.zeros((self._n_dims,))
            for i in range(N):
                if r_ik[k] == k:
                    count += 1
                    accumlator += x[i]
            self._mu[k] = np.divide(accumlator,float(count))

    def get_posterior(self, x):
        """Computes cluster assignment.

        Computes the posterior probability of p(z|x), z is the latent cluster
        variable.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            r_ik(numpy.ndarray): Array containing the cluster assignment of
            each example, dimension (N,).
        """
        N,ndims = x.shape
        r_ik = np.zeros((N,))
        for i in range(N):
            J = np.zeros((self._n_components,))
            for k in range(self._n_components):
                J[k] = (np.linalg.norm(x[i]-self._mu[k]))**2
            r_ik[i] = np.argmin(J)
        return r_ik

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.

        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.

        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """

        self.cluster_label_map = []
        r_ik = self.get_posterior(x)
        N, = r_ik.shape
        for k in range(self._n_components):
            counter = np.zeros(10)
            for i in range(N):
                if r_ik[i] == k:
                    counter[int(y[i])] += 1
            self.cluster_label_map.append(np.amax(counter)) 

    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
                x, dimension (N,)
        """

        r_ik = self.get_posterior(x)
        y_hat = []
        return np.array(y_hat)

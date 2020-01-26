import numpy as np
from scipy.linalg.blas import sgemm
import numexpr as ne
from numba import jit


def rbf(X, p, q, gamma=None):
    """A function calculating rbf kernel based projection of data-points in matrix X
            X : array of shape=(n_examples, n_features),
                should be 2-dimensional, even if it consists only of one data-point!
            p : array, first data-point used for projection
            q : array, second data-point used for projection
            gamma: float, default None, gamma in rbf computation
                If None, defaults to 1 / num_features
        Returns
        ----------
            projection : array of shape=(n_examples,) with distance-based projection values
        Note
        ----------

    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    return np.exp(-gamma * (ne.evaluate("(X - q) ** 2").sum(1))) -\
           np.exp(-gamma * (ne.evaluate("(X - p) ** 2").sum(1)))


def sqeuclidean(X, p, q):
    """A function calculating squared euclidean distance based projection of data-points in matrix X
        Parameters
        ----------
            X : array of shape=(n_examples, n_features),
                should be 2-dimensional, even if it consists only of one data-point!
            p : array, first data-point used for projection
            q : array, second data-point used for projection

        Returns
        ----------
            projection : array of shape=(n_examples,) with distance-based projection values

        Notes
        ----------
            (X-q)^2 - (X-p)^2 = X^2 - 2X*q + q^2 - (X^2 - 2X*p + p^2) =
                                2X(-q + p) + q^2 - p^2
                                Note that q^2 - p^2 is independent of X, so we can write:
                                X(p - q)
    """

    return np.dot(X, p - q)


def euclidean(X, p, q):
    """A function calculating euclidean distance based projection of data-points in matrix X
        Parameters
        ----------
            X : array of shape=(n_examples, n_features),
                should be 2-dimensional, even if it consists only of one data-point!
            p : array, first data-point used for projection
            q : array, second data-point used for projection

        Returns
        ----------
            projection : array of shape=(n_examples,) with distance-based projection values

        Notes
        ----------
            the other option is : np.linalg.norm(X - q, axis=1) - np.linalg.norm(X - p, axis=1)
    """

    return np.sqrt(ne.evaluate("(X - q) ** 2").sum(1)) - np.sqrt(ne.evaluate("(X - p) ** 2").sum(1))


def dot_product(X, p, q):
    """A function calculating dot product distance based projection of data-points in matrix X
        Parameters
        ----------
            X : array of shape=(n_examples, n_features),
                should be 2-dimensional, even if it consists only of one data-point!
            p : array, first data-point used for projection
            q : array, second data-point used for projection

        Returns
        ----------
            projection : array of shape=(n_examples,) with distance-based projection values

        Notes
        ----------
            the other option is : sgemm(alpha=1.0, a=X, b=X, trans_b=True)
    """

    return np.dot(X, q-p)

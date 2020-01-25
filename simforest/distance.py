import numpy as np
from scipy.linalg.blas import sgemm
import numexpr as ne
from numba import jit


def rbf_core(X, gamma):
    """A function calculating rbf kernel, returning the whole distance matrix.
        The same as sklearn.metrics.pairwise.rbf_kernel, but a bit faster
        Parameters
        ----------
            X : matrix, with rows representing vectors for which the distance is to be calculated
            gamma: float, gamma in rbf computation
        Returns
        ----------
            result : array of shape (X.shape[0], X.shape[0]), matrix with rbf distances

    """
    X_norm = -gamma * np.einsum('ij,ij->i', X, X)
    return ne.evaluate('exp(A + B + C)', {
        'A': X_norm[:, None],
        'B': X_norm[None, :],
        'C': sgemm(alpha=2.0 * gamma, a=X, b=X, trans_b=True),
        'g': gamma,
    })


def rbf(x, y, gamma=None):
    """A function calculating rbf kernel, following the sim_function convention
        Parameters
        ----------
            x : array, first data-point
            y : array, second data-point
            gamma: float, default None, gamma in rbf computation
                If None, defaults to 1 / num_features
        Returns
        ----------
            result : rbf kernel distance between the data-points

    """
    if gamma is None:
        gamma = 1.0 / len(x)

    X = np.vstack([x, y])
    result = rbf_core(X, gamma)[1][0]

    return result


def euclidean(X, p, q):
    """A function calculating euclidean distance based projection of data-points in matrix X
        Parameters
        ----------
            X : array of shape=(n_examples, n_features),
                should be 2-dimensional, even if it consists only of one data-point!
            p : array, first data-point used for projection
            q : array, second data-point used for projection
            gamma: float, default None, gamma in rbf computation
                If None, defaults to 1 / num_features
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
            gamma: float, default None, gamma in rbf computation
                If None, defaults to 1 / num_features
        Returns
        ----------
            projection : array of shape=(n_examples,) with distance-based projection values

        Notes
        ----------
            the other option is : sgemm(alpha=1.0, a=X, b=X, trans_b=True)
    """

    return np.dot(X, q) - np.dot(X, p)

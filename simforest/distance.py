import numpy as np
from scipy.linalg.blas import sgemm
import numexpr as ne


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
                If None, defaults to 1.0 / len(x)
        Returns
        ----------
            result : rbf kernel distance between the data-points

    """
    if gamma is None:
        gamma = 1.0 / len(x)

    X = np.vstack([x, y])
    result = rbf_core(X, gamma)[1][0]

    return result

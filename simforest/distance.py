import numpy as np
from scipy.linalg.blas import sgemm
import numexpr as ne


def rbf(X, gamma):
    X_norm = -gamma * np.einsum('ij,ij->i', X, X)
    return ne.evaluate('exp(A + B + C)', {
        'A': X_norm[:, None],
        'B': X_norm[None, :],
        'C': sgemm(alpha=2.0 * gamma, a=X, b=X, trans_b=True),
        'g': gamma,
    })

import pytest
from pytest import approx
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from simforest.distance import rbf


def test_rbf():
    x1 = np.array([1, 0, 0])
    x2 = np.array([1, 0, 0])
    rbf1 = rbf(np.vstack([x1, x2]), 1 / len(x1))[1][0]
    rbf2 = rbf_kernel(np.vstack([x1, x2]))[1][0]
    assert rbf1 == approx(rbf2)

    x1 = np.array([1, 1, 1])
    x2 = np.array([1, 2, 2])
    rbf1 = rbf(np.vstack([x1, x2]), 1 / len(x1))[1][0]
    rbf2 = rbf_kernel(np.vstack([x1, x2]))[1][0]
    assert rbf1 == approx(rbf2)

    x1 = np.array([0.1, 0.01, 1.5])
    x2 = np.array([2.1, 0.82, 2.15])
    rbf1 = rbf(np.vstack([x1, x2]), 1 / len(x1))[1][0]
    rbf2 = rbf_kernel(np.vstack([x1, x2]))[1][0]
    assert rbf1 == approx(rbf2)

    x1 = np.array([0.01, 0.001, 0.000015])
    x2 = np.array([0.21, 0.082, 2.15])
    rbf1 = rbf(np.vstack([x1, x2]), 1 / len(x1))[1][0]
    rbf2 = rbf_kernel(np.vstack([x1, x2]))[1][0]
    assert rbf1 == approx(rbf2)

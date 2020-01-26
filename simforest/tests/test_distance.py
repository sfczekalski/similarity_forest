import pytest
from pytest import approx
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from simforest.distance import rbf, sqeuclidean
from simforest._distance import rbf_sequential as crbf
from scipy.spatial.distance import sqeuclidean as refsqeuclidean


def test_rbf():
    x1 = np.array([1, 0, 0])
    x2 = np.array([1, 0, 0])
    x3 = np.array([1, 0, 0])

    # rbf(x, q) - rbf(x, p)

    rbf1 = rbf(x1.reshape(1, -1), x2, x3)
    rbf2 = rbf_kernel(np.vstack([x1, x3]))[1][0] - rbf_kernel(np.vstack([x1, x2]))[1][0]
    assert rbf1[0] == approx(rbf2)

    x1 = np.array([1, 1, 1])
    x2 = np.array([1, 2, 2])
    x3 = np.array([3, 2, 1])
    rbf1 = rbf(x1.reshape(1, -1), x2, x3)
    rbf2 = rbf_kernel(np.vstack([x1, x3]))[1][0] - rbf_kernel(np.vstack([x1, x2]))[1][0]
    assert rbf1[0] == approx(rbf2)

    x1 = np.array([0.1, 0.01, 1.5])
    x2 = np.array([2.1, 0.82, 2.15])
    x2 = np.array([5.1, 2.82, 3.15])
    rbf1 = rbf(x1.reshape(1, -1), x2, x3)
    rbf2 = rbf_kernel(np.vstack([x1, x3]))[1][0] - rbf_kernel(np.vstack([x1, x2]))[1][0]
    assert rbf1[0] == approx(rbf2)

    x1 = np.array([0.01, 0.001, 0.000015])
    x2 = np.array([0.21, 0.082, 2.15])
    x2 = np.array([7.21, 7.082, 1.15])
    rbf1 = rbf(x1.reshape(1, -1), x2, x3)
    rbf2 = rbf_kernel(np.vstack([x1, x3]))[1][0] - rbf_kernel(np.vstack([x1, x2]))[1][0]
    assert rbf1[0] == approx(rbf2)


def test_crbf():
    x1 = np.array([1, 0, 0], dtype=np.float32)
    x2 = np.array([1, 0, 0], dtype=np.float32)
    rbf1 = crbf(x1, x2)
    rbf2 = rbf_kernel(np.vstack([x1, x2]))[1][0]
    assert rbf1 == approx(rbf2)

    x1 = np.array([1, 1, 1], dtype=np.float32)
    x2 = np.array([1, 2, 2], dtype=np.float32)
    rbf1 = crbf(x1, x2)
    rbf2 = rbf_kernel(np.vstack([x1, x2]))[1][0]
    assert rbf1 == approx(rbf2)

    x1 = np.array([0.1, 0.01, 1.5], dtype=np.float32)
    x2 = np.array([2.1, 0.82, 2.15], dtype=np.float32)
    rbf1 = crbf(x1, x2)
    rbf2 = rbf_kernel(np.vstack([x1, x2]))[1][0]
    assert rbf1 == approx(rbf2)

    x1 = np.array([0.01, 0.001, 0.000015], dtype=np.float32)
    x2 = np.array([0.21, 0.082, 2.15], dtype=np.float32)
    rbf1 = crbf(x1, x2)
    rbf2 = rbf_kernel(np.vstack([x1, x2]))[1][0]
    assert rbf1 == approx(rbf2)


'''def test_squclidean():
    x1 = np.array([1, 0, 0])
    x2 = np.array([1, 0, 0])
    x3 = np.array([1, 0, 0])


    res1 = sqeuclidean(x1.reshape(1, -1), x2, x3)
    res2 = refsqeuclidean(x1, x3) - refsqeuclidean(x1, x2)
    assert res1[0] == approx(res2)

    x1 = np.array([1, 1, 1])
    x2 = np.array([1, 2, 2])
    x3 = np.array([3, 2, 1])
    res1 = sqeuclidean(x1.reshape(1, -1), x2, x3)
    res2 = refsqeuclidean(x1, x3) - refsqeuclidean(x1, x2)
    assert res1[0] == approx(res2)

    x1 = np.array([0.1, 0.01, 1.5])
    x2 = np.array([2.1, 0.82, 2.15])
    x3 = np.array([5.1, 2.82, 3.15])
    res1 = sqeuclidean(x1.reshape(1, -1), x2, x3)
    res2 = refsqeuclidean(x1, x3) - refsqeuclidean(x1, x2)
    assert res1[0] == approx(res2)

    x1 = np.array([0.01, 0.001, 0.000015])
    x2 = np.array([0.21, 0.082, 2.15])
    x3 = np.array([7.21, 7.082, 1.15])
    res1 = sqeuclidean(x1.reshape(1, -1), x2, x3)
    res2 = refsqeuclidean(x1, x3) - refsqeuclidean(x1, x2)
    assert res1[0] == approx(res2)'''

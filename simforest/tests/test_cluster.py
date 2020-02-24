import pytest
import numpy as np
from sklearn.datasets import load_iris, make_blobs
from simforest.cluster import PySimilarityTreeCluster, PySimilarityForestCluster, \
    SimilarityForestCluster
from scipy.special import comb
from sklearn.utils.validation import check_symmetric
from scipy.spatial.distance import squareform


@pytest.fixture
def data():
    return load_iris(return_X_y=True)

'''
def test_sampling_directions():
    model = SimilarityTreeCluster()

    # prepare data, get its bootstrap sample
    X = np.random.rand(100, 4)
    n = X.shape[0]
    idxs = np.random.choice(range(n), n, replace=True)

    # first split point is randomly sampled
    # the second one is chosen such that it is for sure not a copy of first one
    i = np.random.randint(0, n)
    j = model.sample_split_direction(X[idxs], i)
    assert not np.array_equal(X[i], X[j])

    # check the same thing in case of such data
    X = np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [0, 0, 0]])
    i = 0
    j = model.sample_split_direction(X, i)
    assert not np.array_equal(X[i], X[j])

'''


def is_pure(X):
    n = X.shape[0]
    m = X.shape[1]
    pure = 1
    for i in range(n-1):
        for j in range(m):
            if X[i, j] != X[i+1, j]:
                print(f'Catch ya! {i, j}, {i+1, j}')
                pure = 0
                break

    return pure


def test_is_pure():
    X = np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1],
                  [0, 0, 0]])
    assert is_pure(X) == 0

    X = np.ones(shape=(1000, 2))
    assert is_pure(X) == 1


def test_similarity_forest_cluster_output_array_shape(data):
    X, y = data
    model = SimilarityForestCluster()

    model.fit(X)
    assert len(model.estimators_) == model.n_estimators
    assert model.labels_.shape[0] == X.shape[0]
    assert model.distance_matrix_.shape == (X.shape[0], X.shape[0])


'''def test_distance_matrix_symmetric(data):
    X, y = data
    model = SimilarityForestCluster()

    distance_matrix = model.fit_predict(X)
    check_symmetric(squareform(distance_matrix))'''

import pytest
import numpy as np
from sklearn.datasets import load_iris, make_blobs
from simforest.cluster import SimilarityTreeCluster, SimilarityForestCluster
from scipy.special import comb


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_sampling_directions():
    model = SimilarityTreeCluster()

    # prepare data, get its bootstrap sample
    X = np.random.rand(100, 4)
    n = X.shape[0]
    idxs = np.random.choice(range(n), n, replace=True)

    # make sure two different points are always chosen
    i, j = model._sample_directions(np.random.RandomState(), X[idxs])
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
    i, j = model._sample_directions(np.random.RandomState(), X)
    assert not np.array_equal(X[i], X[j])


def test_similarity_tree_cluster_output_array_shape(data):
    X, y = data
    model = SimilarityTreeCluster()

    model.fit(X)


def test_similarity_forest_cluster_output_array_shape(data):
    X, y = data
    model = SimilarityForestCluster()

    model.fit(X)
    assert len(model.estimators_) == model.n_estimators
    assert model.sf_distance(X).shape == (comb(X.shape[0], 2),)

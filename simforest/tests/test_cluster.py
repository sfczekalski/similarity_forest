import pytest
import numpy as np
from sklearn.datasets import load_iris, make_blobs
from simforest.cluster import SimilarityTreeCluster, SimilarityForestCluster
from scipy.special import comb

@pytest.fixture
def data():
    return load_iris(return_X_y=True)


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

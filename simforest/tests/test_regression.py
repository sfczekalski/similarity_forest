import pytest
import numpy as np

from sklearn.datasets import load_boston
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose
from scipy.spatial import distance

from simforest import SimilarityTreeRegressor


@pytest.fixture
def data():
    return load_boston(return_X_y=True)


def test_similarity_tree_regressor_output_array_shape(data):
    X, y = data
    clf = SimilarityTreeRegressor()

    clf.fit(X, y)

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_regressor_attributes_tree(data):
    X, y = data
    clf = SimilarityTreeRegressor()

    clf.fit(X, y)

    assert hasattr(clf, 'is_fitted_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')


def test_default_attribute_value_tree():

    clf = SimilarityTreeRegressor()
    assert clf.random_state == 1
    assert clf.n_directions == 1
    assert clf.sim_function == np.dot


def test_setting_attributes_tree(data):
    X, y = data
    clf = SimilarityTreeRegressor(random_state=42, sim_function=distance.cosine, n_directions=2)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert clf.random_state == 42
    assert clf.sim_function == distance.cosine
    assert clf.n_directions == 2


def test_deterministic_predictions_tree(data):
    X, y = data

    clf1 = SimilarityTreeRegressor(random_state=42)
    clf1.fit(X, y)
    clf2 = SimilarityTreeRegressor(random_state=42)
    clf2.fit(X, y)

    y_pred1 = clf1.predict(X)
    y_pred2 = clf2.predict(X)
    assert_array_equal(y_pred1, y_pred2)


def test_wrong_sim_f_tree():
    with pytest.raises(ValueError) as wrong_sim_f:
        X, y = np.array(['a', 'b', 'c']), np.array([1.0, 0.0, 0.0])
        clf = SimilarityTreeRegressor()
        clf.fit(X, y)
        assert 'Provided similarity function does not apply to input.' in str(wrong_sim_f.value)
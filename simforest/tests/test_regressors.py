import pytest
import numpy as np

from sklearn.datasets import load_boston
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose
from scipy.spatial import distance
from simforest.criterion import find_split_variance, find_split_theil

from simforest import SimilarityTreeRegressor, SimilarityForestRegressor


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


def test_similarity_forest_regressor_output_array_shape(data):
    X, y = data
    clf = SimilarityTreeRegressor()

    clf.fit(X, y)

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_classifier_attributes_forest(data):
    X, y = data
    clf = SimilarityForestRegressor()

    clf.fit(X, y)

    assert hasattr(clf, 'is_fitted_')


def test_default_attribute_value_forest():

    clf = SimilarityForestRegressor()
    assert clf.n_directions == 1
    assert clf.sim_function == np.dot


def test_setting_attributes_forest(data):
    X, y = data
    clf = SimilarityForestRegressor(random_state=42, sim_function=distance.cosine, n_directions=2)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert clf.random_state == 42
    assert clf.sim_function == distance.cosine
    assert clf.n_directions == 2


def test_deterministic_predictions_forest(data):
    X, y = data

    clf1 = SimilarityForestRegressor(random_state=42)
    clf1.fit(X, y)
    clf2 = SimilarityForestRegressor(random_state=42)
    clf2.fit(X, y)

    y_pred1 = clf1.predict(X)
    y_pred2 = clf2.predict(X)
    assert_array_equal(y_pred1, y_pred2)


def test_wrong_sim_f_forest():

    with pytest.raises(ValueError) as wrong_sim_f:
        X, y = np.array(['a', 'b', 'c']), np.array([1, 0, 0])
        clf = SimilarityForestRegressor()
        clf.fit(X, y)
        assert 'Provided similarity function does not apply to input.' in str(wrong_sim_f.value)


def test_number_of_tree_leaves_in_apply(data):
    X, y = data
    clf = SimilarityTreeRegressor()
    clf.fit(X, y)

    assert (np.unique(clf.apply(X)).size == clf.get_n_leaves())


def test_number_of_tree_in_forest_leaves_in_apply(data):
    X, y = data
    clf = SimilarityForestRegressor()
    clf.fit(X, y)
    apply_result = clf.apply(X)

    assert np.unique(apply_result[:, 0]).size == clf.estimators_[0].get_n_leaves()


def test_forest_apply_result_shape(data):
    X, y = data
    clf = SimilarityForestRegressor()
    clf.fit(X, y)
    apply_result = clf.apply(X)

    assert apply_result.shape == (X.shape[0], clf.n_estimators)


def test_var_split():
    y = np.array([1.5, 1.5, 0.0], dtype=np.float32)
    s = np.array([1., 2., 3.], dtype=np.float32)
    i, impurity = find_split_variance(y, s, np.int32(2))
    assert i == 1
    assert impurity == 0.0

import pytest
import numpy as np

from sklearn.datasets import load_iris, make_blobs
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose
from scipy.spatial import distance

from simforest import SimilarityTreeClassifier, SimilarityForestClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_similarity_tree_classifier_output_array_shape(data):
    X, y = data
    clf = SimilarityTreeClassifier()

    clf.fit(X, y)

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_classifier_attributes_tree(data):
    X, y = data
    clf = SimilarityTreeClassifier()

    clf.fit(X, y)

    assert hasattr(clf, 'is_fitted_')
    assert hasattr(clf, 'classes')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')


def test_default_attribute_value_tree():

    clf = SimilarityTreeClassifier()
    assert clf.random_state == 1
    assert clf.n_directions == 1
    assert clf.sim_function == np.dot


def test_setting_attributes_tree(data):
    X, y = data
    clf = SimilarityTreeClassifier(random_state=42, sim_function=distance.cosine, n_directions=2)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert clf.random_state == 42
    assert clf.sim_function == distance.cosine
    assert clf.n_directions == 2


def test_deterministic_predictions_tree():
    X, y = make_blobs(n_samples=300, centers=[(0, 0), (1, 1)], random_state=42)

    clf1 = SimilarityTreeClassifier(random_state=42)
    clf1.fit(X, y)
    clf2 = SimilarityTreeClassifier(random_state=42)
    clf2.fit(X, y)

    y_pred1 = clf1.predict(X)
    y_pred2 = clf2.predict(X)
    assert_array_equal(y_pred1, y_pred2)


def test_log_probabilities_tree(data):
    X, y = data
    clf = SimilarityTreeClassifier()
    clf.fit(X, y)
    preds = clf.predict_proba(X)
    log_preds = clf.predict_log_proba(X)

    assert_allclose(log_preds, np.log(preds+1e-10))


def test_pure_node():
    X = np.ndarray(shape=(2,2), dtype=float, order='F')
    y = np.zeros(shape=(2,), dtype=np.int64)
    clf = SimilarityTreeClassifier()
    clf.fit(X, y)
    assert clf._is_leaf == True


def test_wrong_sim_f_tree():
    with pytest.raises(ValueError) as wrong_sim_f:
        X, y = np.array(['a', 'b', 'c']), np.array([1, 0, 0])
        clf = SimilarityTreeClassifier()
        clf.fit(X, y)
        assert 'Provided similarity function does not apply to input.' in str(wrong_sim_f.value)


def test_probability_values_tree(data):
    X, y = data
    clf = SimilarityTreeClassifier()
    clf.fit(X, y)
    preds = clf.predict_proba(X)

    assert_allclose(np.sum(preds, axis=1), np.ones(shape=y.shape))

def test_similarity_forest_classifier_output_array_shape(data):
    X, y = data
    clf = SimilarityForestClassifier()

    clf.fit(X, y)

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_classifier_attributes_forest(data):
    X, y = data
    clf = SimilarityForestClassifier()

    clf.fit(X, y)

    assert hasattr(clf, 'is_fitted_')
    assert hasattr(clf, 'classes')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')


def test_default_attribute_value_forest():

    clf = SimilarityForestClassifier()
    assert clf.random_state == 1
    assert clf.n_directions == 1
    assert clf.sim_function == np.dot


def test_setting_attributes_forest(data):
    X, y = data
    clf = SimilarityForestClassifier(random_state=42, sim_function=distance.cosine, n_directions=2)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert clf.random_state == 42
    assert clf.sim_function == distance.cosine
    assert clf.n_directions == 2


def test_deterministic_predictions_forest():
    X, y = make_blobs(n_samples=300, centers=[(0, 0), (1, 1)], random_state=42)

    clf1 = SimilarityForestClassifier(random_state=42)
    clf1.fit(X, y)
    clf2 = SimilarityForestClassifier(random_state=42)
    clf2.fit(X, y)

    y_pred1 = clf1.predict(X)
    y_pred2 = clf2.predict(X)
    assert_array_equal(y_pred1, y_pred2)


def test_log_probabilities_forest(data):
    X, y = data
    clf = SimilarityForestClassifier()
    clf.fit(X, y)
    preds = clf.predict_proba(X)
    log_preds = clf.predict_log_proba(X)

    assert_allclose(log_preds, np.log(preds+1e-10))


def test_wrong_sim_f_forest():

    with pytest.raises(ValueError) as wrong_sim_f:
        X, y = np.array(['a', 'b', 'c']), np.array([1, 0, 0])
        clf = SimilarityForestClassifier()
        clf.fit(X, y)
        assert 'Provided similarity function does not apply to input.' in str(wrong_sim_f.value)

#test probabilty values - sum axis 1 == 1


def test_probability_values_forest(data):
    X, y = data
    clf = SimilarityForestClassifier()
    clf.fit(X, y)
    preds = clf.predict_proba(X)

    assert_allclose(np.sum(preds, axis=1), np.ones(shape=y.shape))

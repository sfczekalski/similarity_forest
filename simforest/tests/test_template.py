import pytest
import numpy as np

from sklearn.datasets import load_iris, make_blobs
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose
from scipy.spatial import distance

from simforest import SimilarityTreeClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)

'''
def test_similarity_tree_predictions(data):
    clf = SimilarityTreeClassifier()

    clf.fit(*data)

    X = data[0]
    y_pred = clf.predict(X)
    assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))
    assert y_pred.shape == (X.shape[0],)
'''

def test_similarity_tree_classifier_output_array_shape(data):
    X, y = data
    clf = SimilarityTreeClassifier()

    clf.fit(X, y)

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_classifier_attributes(data):
    X, y = data
    clf = SimilarityTreeClassifier()

    clf.fit(X, y)

    assert hasattr(clf, 'is_fitted_')
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')


def test_default_attribute_value():

    clf = SimilarityTreeClassifier()
    assert clf.random_state == 1
    assert clf.n_directions == 1
    assert clf._sim_function == np.dot


def test_setting_attributes(data):
    X, y = data
    clf = SimilarityTreeClassifier(random_state=42, sim_function=distance.cosine, n_directions=2)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert clf.random_state == 42
    assert clf._sim_function == distance.cosine
    assert clf.n_directions == 2


def test_deterministic_predictions():
    X, y = make_blobs(n_samples=300, centers=[(0, 0), (1, 1)], random_state=42)

    clf1 = SimilarityTreeClassifier(random_state=42)
    clf1.fit(X, y)
    clf2 = SimilarityTreeClassifier(random_state=42)
    clf2.fit(X, y)

    y_pred1 = clf1.predict(X)
    y_pred2 = clf2.predict(X)
    assert_array_equal(y_pred1, y_pred2)


def test_raises_value_error_wron_sim_fun(data):

    def wrong_sim_f(x1, x2):
        return 'wrong'

    with pytest.raises(ValueError, match='Provided similarity function does not apply to input.'):
        X, y = data
        clf = SimilarityTreeClassifier(sim_function=wrong_sim_f)
        clf.fit(X, y)
        y_pred = clf.predict(X)


def test_wrong_class_prob_value_range():
    X, y = make_blobs(n_samples=300, centers=[(0, 0), (1, 1)], random_state=42)
    y += 1
    with pytest.raises(ValueError, match='Wrong node class probability value.'):
        clf = SimilarityTreeClassifier()
        clf.fit(X, y)


def test_pure_node():
    X = np.ndarray(shape=(2,2), dtype=float, order='F')
    y = np.zeros(shape=(2,), dtype=np.int64)
    clf = SimilarityTreeClassifier()
    clf.fit(X, y)
    assert clf._is_leaf == True

import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from simforest import SimilarityTreeClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_template_estimator_predictions(data):
    clf = SimilarityTreeClassifier()

    clf.fit(*data)

    X = data[0]
    y_pred = clf.predict_proba(X)
    assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.float32))
    assert y_pred.shape == (X.shape[0],)


def test_classifier_attributes(data):
    X, y = data
    clf = SimilarityTreeClassifier()

    clf.fit(X, y)

    assert clf.demo_param == 'demo_param'
    assert hasattr(clf, 'is_fitted_')
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')

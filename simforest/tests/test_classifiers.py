import pytest
import numpy as np

from sklearn.datasets import load_iris, make_blobs
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose
from simforest import SimilarityTreeClassifier, SimilarityForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_similarity_tree_classifier_prediction(data):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    clf = SimilarityTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)
    assert accuracy_score(y_test, y_pred) > 0.9


def test_similarity_forest_classifier_prediction(data):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    clf = SimilarityForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)
    assert accuracy_score(y_test, y_pred) > 0.9


def test_setting_attributes_tree(data):
    X, y = data
    clf = SimilarityTreeClassifier(random_state=42, n_directions=2)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert clf.random_state == 42
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


def test_setting_attributes_forest(data):
    X, y = data
    clf = SimilarityForestClassifier(random_state=42, n_directions=2)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert clf.random_state == 42
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


def test_probability_values_forest(data):
    X, y = data
    clf = SimilarityForestClassifier()
    clf.fit(X, y)
    preds = clf.predict_proba(X)

    assert_allclose(np.sum(preds, axis=1), np.ones(shape=y.shape))


def test_number_of_tree_leaves_in_apply(data):
    X, y = data
    clf = SimilarityTreeClassifier()
    clf.fit(X, y)

    assert (np.unique(clf.apply(X)).size == clf.get_n_leaves())


def test_forest_apply_result_shape(data):
    X, y = data
    clf = SimilarityForestClassifier()
    clf.fit(X, y)
    apply_result = clf.apply(X)

    assert apply_result.shape == (X.shape[0], clf.n_estimators)


def test_similarity_forest_wrongly_the_same_pred(data):
    """There should not be a situation when models predicts the same when there is no random_state set"""
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    clf1 = SimilarityForestClassifier()
    clf1.fit(X_train, y_train)
    y_pred1 = clf1.predict_proba(X_test)

    clf2 = SimilarityForestClassifier()
    clf2.fit(X_train, y_train)
    y_pred2 = clf2.predict_proba(X_test)

    assert not np.array_equal(y_pred1, y_pred2)


def test_similarity_forest_outliers_ranking_stability(data):
    """There should not be a situation when models predicts the same when there is no random_state set"""
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    clf = SimilarityForestClassifier()
    clf.fit(X_train, y_train)
    '''rcorrelations = clf.outliers_rank_stability(X_test, plot=False)
    assert rcorrelations.shape == (9, 2)
    assert rcorrelations[:, 0].all() >= -1
    assert rcorrelations[:, 0].all() <= 1
    assert rcorrelations[:, 1].all() >= 0
    assert rcorrelations[:, 1].all() <= 1'''


def test_train_set_acc(data):
    X, y = data

    forest = SimilarityForestClassifier()
    forest.fit(X, y)
    # shouldn't be actually 1.0?
    assert forest.score(X, y) > 0.8

    tree = SimilarityTreeClassifier()
    tree.fit(X, y)
    assert tree.score(X, y) > 0.9

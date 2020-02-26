import pytest
import numpy as np
from simforest.isolation_simforest import IsolationSimilarityForest
from examples.outliers.outliers_datasets import get_kddcup99_http
from sklearn.metrics import roc_auc_score
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.estimator_checks import _pairwise_estimator_convert_X
from sklearn.model_selection import ShuffleSplit
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils._testing import assert_allclose_dense_sparse


def test_isolation_forest():
    X_train, X_test, y_train, y_test, _ = get_kddcup99_http()
    sf = IsolationSimilarityForest()
    sf.fit(X_train, y_train)
    sf_pred = sf.decision_function(X_test)
    assert sf_pred.shape == (X_test.shape[0],)
    assert roc_auc_score(y_test, sf_pred) > 0.8

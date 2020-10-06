from simforest.outliers.isolation_simforest import IsolationSimilarityForest
from examples.outliers.outliers_datasets import get_kddcup99_http
from sklearn.metrics import roc_auc_score


def test_isolation_forest():
    X_train, X_test, y_train, y_test, _ = get_kddcup99_http()
    sf = IsolationSimilarityForest()
    sf.fit(X_train, y_train)
    sf_pred = sf.decision_function(X_test)
    assert sf_pred.shape == (X_test.shape[0],)
    assert roc_auc_score(y_test, sf_pred) > 0.8

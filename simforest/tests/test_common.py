import pytest
from sklearn.utils.estimator_checks import check_estimator
from simforest import SimilarityTreeClassifier, SimilarityForestClassifier, SimilarityTreeRegressor, \
    SimilarityForestRegressor
from simforest.outliers.isolation_simforest import IsolationSimilarityTree, IsolationSimilarityForest
from sklearn.utils.estimator_checks import parametrize_with_checks


@pytest.mark.parametrize("Estimator", [SimilarityTreeClassifier, SimilarityForestClassifier,
                                       SimilarityTreeRegressor, SimilarityForestRegressor])
def test_all_estimators(Estimator):
    return check_estimator(Estimator)


# This method runs all test independently, not sequentially, and reports all fails
# For now all the tests are passed, except for the one related to pickling
# SimilarityForestCluster,
@parametrize_with_checks([IsolationSimilarityTree, IsolationSimilarityForest])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)

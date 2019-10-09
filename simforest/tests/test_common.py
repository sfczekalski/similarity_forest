import pytest

from sklearn.utils.estimator_checks import check_estimator

from simforest import SimilarityTreeRegressor


@pytest.mark.parametrize(
    "Estimator", [SimilarityTreeRegressor]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)

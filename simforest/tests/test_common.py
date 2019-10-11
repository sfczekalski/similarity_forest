import pytest

from sklearn.utils.estimator_checks import check_estimator

from simforest import SimilarityTreeRegressor, SimilarityForestRegressor


@pytest.mark.parametrize(
    "Estimator", [SimilarityTreeRegressor, SimilarityForestRegressor]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)

import pytest

from sklearn.utils.estimator_checks import check_estimator

from simforest import SimilarityTreeClassifier


@pytest.mark.parametrize(
    "Estimator", [SimilarityTreeClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)

import pytest

from sklearn.utils.estimator_checks import check_estimator

from simforest import SimilarityTreeClassifier


@pytest.mark.parametrize(
    "SimilarityTreeClassifier", [SimilarityTreeClassifier]
)
def test_all_estimators(SimilarityTreeClassifier):
    return check_estimator(SimilarityTreeClassifier)

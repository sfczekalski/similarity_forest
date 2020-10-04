from ._classes import SimilarityTreeClassifier, SimilarityForestClassifier, \
    SimilarityTreeRegressor, SimilarityForestRegressor

from .cluster import SimilarityForestCluster
from ._version import __version__

__all__ = ['SimilarityTreeClassifier', 'SimilarityForestClassifier',
           'SimilarityTreeRegressor', 'SimilarityForestRegressor', 'SimilarityForestCluster',
           '__version__']

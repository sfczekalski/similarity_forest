import numpy as np
from criterion import find_split_variance, find_split_theil, find_split_atkinson, find_split_index_gini
from rcriterion import gini_index
from sklearn.preprocessing import LabelEncoder


def find_split(X, y, p, q, criterion, sim_function):
    """ Find split among direction drew on pair of data-points
        Parameters
        ----------
            X : all data-points
            y : output vector
            p : first data-point used for drawing direction of the split
            q : second data-point used for drawing direction of the split
            criterion : criterion, criterion to be minimized when finding for optimal split
            sim_function : function used to measure similarity between data-points

        Returns
        -------
            impurity : impurity induced by the split (value of criterion function)
            split_point : split threshold
            similarities : array of shape (n_samples,), values of similarity-values based projection
    """
    similarities = sim_function(X, p, q)
    indices = sorted([i for i in range(len(y)) if not np.isnan(similarities[i])],
                     key=lambda x: similarities[x])
    y = y[indices]
    n = len(y)

    if criterion == 'variance':
        i, impurity = find_split_variance(y.astype(np.float32),
                                          similarities[indices].astype(np.float32),
                                          np.int32(n - 1))

    elif criterion == 'theil':
        i, impurity = find_split_theil(y[indices].astype(np.float32),
                                       similarities[indices].astype(np.float32),
                                       np.int32(n - 1))

    elif criterion == 'atkinson':
        i, impurity = find_split_atkinson(y[indices].astype(np.float32),
                                          similarities[indices].astype(np.float32),
                                          np.int32(n - 1))

    elif criterion == 'gini':
        if y.dtype != int:
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)

        y = np.array(y[indices], dtype=np.int32)
        classes = np.unique(y).astype(np.int32)

        i, impurity = find_split_index_gini(y[indices], np.int32(n - 1), classes)

    split_point = (similarities[indices[i]] + similarities[indices[i + 1]]) / 2

    return impurity, split_point, similarities

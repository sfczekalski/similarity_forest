cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float gini_index(int split_index, int [:] y, int [:] classes, int len_y):
    """Calculate Gini index on a given array, at given index
        Parameters
        ----------
            split_index : np.int32, index of split
            y : numpy array of np.int32, array of points' classes
            classes : numpy array of np.int32, array of unique classes
            len_y : np.int32, length of y
        Returns
        ----------
            gini : float, Gini index for current split
    """

    cdef int [:] left_partition = y[:split_index]
    cdef int [:] right_partition = y[split_index:]
    #cdef int len_y = y.shape[0]
    cdef Py_ssize_t len_left_partition = left_partition.shape[0]
    cdef long len_left_partition2 = len_left_partition * len_left_partition
    cdef Py_ssize_t len_right_partition = len_y - len_left_partition
    cdef long len_right_partition2 = len_right_partition * len_right_partition

    cdef Py_ssize_t n_classes = classes.shape[0]

    cdef float left_gini = 0.0
    cdef float right_gini = 0.0
    cdef int class_count = 0

    # calc left gini
    for c in range(n_classes):
        for i in range(len_left_partition):
            if left_partition[i] == classes[c]:
                class_count = class_count + 1
        left_gini = left_gini + class_count * class_count
        class_count = 0

    left_gini = left_gini / len_left_partition2
    left_gini = 1 - left_gini

    # calc right gini
    for c in range(n_classes):
        for i in range(len_right_partition):
            if right_partition[i] == classes[c]:
                class_count = class_count + 1
        right_gini = right_gini + class_count * class_count
        class_count = 0

    right_gini = right_gini / len_right_partition2
    right_gini = 1 - right_gini

    # float length is necessary, cython int division below leads to 0. Try compiler directive #cython: language_level=3
    cdef float flen_left_partition = left_partition.shape[0]
    cdef float left_proportion = flen_left_partition / len_y

    return left_proportion * left_gini  + (1.0 - left_proportion) * right_gini

@cython.boundscheck(False)
@cython.wraparound(False)
def find_split_index(int [:] y, int max_range, int [:] classes):
    """This is a function calculating optimal split point, and impurity of partitions after splitting
            Parameters
            ---------
                y : numpy array of type np.int32, array of objects' labels, order by objects' similarity
                max_range : np.int32, should be length of y array - 1, so at least one point is left in right partition
                classes : numpy array of type np.int32, all unique labels
            Returns
            ---------
                best_split_idx : int, index of element at which optimal split should be performed
                best_impurity : float, impurity after split
    """

    cdef float best_impurity = 1.0
    cdef int best_split_idx = -1
    cdef float curr_impurity = 1.0
    cdef int len_y = y.shape[0]

    cdef int i = 0
    while i < max_range:
        curr_impurity = gini_index(i+1, y, classes, len_y)
        if curr_impurity < best_impurity:
            best_impurity = curr_impurity
            best_split_idx = i

        i += 1

    return best_split_idx, best_impurity

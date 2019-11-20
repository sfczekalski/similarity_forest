import numpy as np
cimport cython
DTYPE = np.intc

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float gini_index(int split_index, int [:] y, int [:] classes):
    cdef int [:] left_partition = y[:split_index]
    cdef int [:] right_partition = y[split_index:]
    cdef Py_ssize_t len_y = y.shape[0]
    cdef Py_ssize_t len_left_partition = left_partition.shape[0]
    cdef Py_ssize_t len_right_partition = len_y - len_left_partition

    cdef Py_ssize_t n_classes = classes.shape[0]

    cdef float left_gini = 0.0
    cdef float right_gini = 0.0
    cdef int class_count = 0

    # calc left gini
    for c in range(n_classes):
        for i in range(len_left_partition):
            if left_partition[i] == classes[c]:
                class_count = class_count + 1
        left_gini = left_gini + (class_count * class_count) / (len_left_partition * len_left_partition)
        class_count = 0

    left_gini = 1 - left_gini

    # calc right gini
    for c in range(n_classes):
        for i in range(len_right_partition):
            if right_partition[i] == classes[c]:
                class_count = class_count + 1
        right_gini = right_gini + (class_count * class_count) / (len_right_partition * len_right_partition)
        class_count = 0

    right_gini = 1 - right_gini

    cdef float left_prop = len_left_partition / len_y
    return left_prop * left_gini + (1.0 - left_prop) * right_gini

@cython.boundscheck(False)
@cython.wraparound(False)
def find_split_index(int [:] y, int max_range, int [:] classes):
    """This is a function calculating optimal split point, and impurity of partitions after splitting"""

    cdef float best_impurity = 1.0
    cdef int best_split_idx = -1
    cdef float curr_impurity = 1.0

    for i in range(max_range):
        curr_impurity = gini_index(i+1, y, classes)
        if curr_impurity < best_impurity:
                best_impurity = curr_impurity
                best_split_idx = i


    return best_split_idx, best_impurity

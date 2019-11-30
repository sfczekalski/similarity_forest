cimport cython
from libc.math cimport log

cdef extern from "math.h":
    float INFINITY


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float var_agg(int cnt, float sum1, float sum2):
    cdef float result = (sum2/(cnt + 0.01)) - (sum1/(cnt + 0.01))**2
    #assert result > 0
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float sum_squared_array(float [:] y, int n):

    cdef int i = 0
    cdef float result = 0.0
    while i < n:
        result += y[i] ** 2
        i += 1

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float sum_array(float [:] y, int n):

    cdef int i = 0
    cdef float result = 0.0
    while i < n:
        result += y[i]
        i += 1

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float weighted_variance(int split_index, float [:] y, int len_y):
    cdef float [:] left_partition = y[:split_index]
    cdef float [:] right_partition = y[split_index:]
    cdef int len_left_partition = left_partition.shape[0]
    cdef int len_right_partition = len_y - len_left_partition

    cdef float flen_left_partition = left_partition.shape[0]
    cdef float left_proportion = flen_left_partition / len_y

    return left_proportion * variance(left_partition, len_left_partition) + \
           (1.0 - left_proportion) * variance(right_partition, len_right_partition)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float variance(float [:] y, int array_size):
    assert array_size >= 1

    cdef float array_sum = 0.0
    cdef int i = 0

    while i < array_size:
        array_sum = array_sum + y[i]
        i = i + 1

    cdef float mean = array_sum / array_size

    cdef float squared_diff = 0.0
    i = 0
    while i < array_size:
        squared_diff += (y[i] - mean) ** 2
        i = i + 1

    cdef float result = squared_diff / array_size
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def find_split_variance_old(float [:] y, float [:] s, int max_range):
    """This is a function calculating optimal split point, and impurity of partitions after splitting
            Parameters
            ---------
                y : numpy array of type np.int32, array of objects' labels, order by objects' similarity
                max_range : np.int32, should be length of y array - 1, so at least one point is left in right partition
            Returns
            ---------
                best_split_idx : int, index of element at which optimal split should be performed
                best_impurity : float, impurity after split
    """

    cdef float best_impurity = INFINITY
    cdef int best_split_idx = -1
    cdef float curr_impurity = INFINITY
    cdef int len_y = y.shape[0]

    cdef int i = 0
    while i < max_range:
        if s[i]==s[i+1]:
            i += 1
            continue
        curr_impurity = weighted_variance(i+1, y, len_y)
        if curr_impurity < best_impurity:
            best_impurity = curr_impurity
            best_split_idx = i

        i += 1

    return best_split_idx, best_impurity


@cython.boundscheck(False)
@cython.wraparound(False)
def find_split_variance(float [:] y, float [:] s, int max_range):
    """This is a function calculating optimal split point, and impurity of partitions after splitting
            Parameters
            ---------
                y : numpy array of type np.int32, array of objects' labels, order by objects' similarity
                max_range : np.int32, should be length of y array - 1, so at least one point is left in right partition
            Returns
            ---------
                best_split_idx : int, index of element at which optimal split should be performed
                best_impurity : float, impurity after split
    """

    cdef int len_y = y.shape[0]
    cdef int best_split_idx = cfind_split_index_var(y, s, max_range, len_y)
    cdef float best_impurity = weighted_variance(best_split_idx+1, y, len_y)

    return best_split_idx, best_impurity


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int cfind_split_index_var(float [:] y, float [:] s, int max_range, int len_y):
    """This is a function calculating optimal split point, and impurity of partitions after splitting
            Parameters
            ---------
                y : numpy array of type np.int32, array of objects' labels, order by objects' similarity
                max_range : np.int32, should be length of y array - 1, so at least one point is left in right partition
            Returns
            ---------
                best_split_idx : int, index of element at which optimal split should be performed
                best_impurity : float, impurity after split
    """

    cdef float best_impurity = float('inf')
    cdef int best_split_idx = -1
    cdef float curr_impurity = 0.0

    cdef int rhs_cnt = len_y
    cdef float rhs_sum = sum_array(y, rhs_cnt)
    cdef float rhs_sum2 = sum_squared_array(y, rhs_cnt)
    cdef int lhs_cnt = 0
    cdef float lhs_sum = 0.0
    cdef float lhs_sum2 = 0.0
    cdef float lhs_var = 0.0
    cdef float rhs_var = 0.0

    cdef float flen_left_partition = 0.0
    cdef float left_proportion = 0.0

    cdef int i = 0
    while i < max_range:
        lhs_cnt += 1
        rhs_cnt -= 1

        flen_left_partition += 1.0
        left_proportion = flen_left_partition / len_y

        lhs_sum += y[i]
        rhs_sum -= y[i]
        lhs_sum2 += (y[i] ** 2)
        rhs_sum2 -= (y[i] ** 2)
        if s[i]==s[i+1]:
            i += 1
            continue
        lhs_var = var_agg(lhs_cnt, lhs_sum, lhs_sum2)
        rhs_var = var_agg(rhs_cnt, rhs_sum, rhs_sum2)
        curr_impurity = lhs_var * left_proportion + rhs_var * (1.0 - left_proportion)
        if curr_impurity < best_impurity:
            best_impurity = curr_impurity
            best_split_idx = i

        i = i + 1

    return best_split_idx


@cython.boundscheck(False)
@cython.wraparound(False)
def find_split_theil(float [:] y, float [:] s, int max_range):
    """This is a function calculating optimal split point, and impurity of partitions after splitting
            Parameters
            ---------
                y : numpy array of type np.int32, array of objects' labels, order by objects' similarity
                max_range : np.int32, should be length of y array - 1, so at least one point is left in right partition
            Returns
            ---------
                best_split_idx : int, index of element at which optimal split should be performed
                best_impurity : float, impurity after split
    """

    cdef float best_impurity = INFINITY
    cdef int best_split_idx = -1
    cdef float curr_impurity = INFINITY
    cdef int len_y = y.shape[0]

    cdef int i = 0
    while i < max_range:
        if s[i]==s[i+1]:
            i += 1
            continue
        curr_impurity = theil_index(i+1, y, len_y)
        if curr_impurity < best_impurity:
            best_impurity = curr_impurity
            best_split_idx = i

        i += 1

    return best_split_idx, best_impurity


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float theil_index(int split_index, float [:] y, int len_y):
    cdef float [:] left_partition = y[:split_index]
    cdef float [:] right_partition = y[split_index:]
    cdef int len_left_partition = left_partition.shape[0]
    cdef int len_right_partition = len_y - len_left_partition

    cdef float flen_left_partition = left_partition.shape[0]
    cdef float left_proportion = flen_left_partition / len_y

    return left_proportion * theil(left_partition, len_left_partition) + \
           (1.0 - left_proportion) * theil(right_partition, len_right_partition)



cdef float theil(float [:] y, int array_size):

    cdef float array_sum = 0.0
    cdef int i = 0
    while i < array_size:
        array_sum = array_sum + y[i]
        i = i + 1

    cdef float mean = array_sum / array_size


    cdef float theil = 0.0
    cdef float t = 0.0
    i = 0
    while i < array_size:
        t = y[i] / mean
        theil = theil + t * log(t)
        i = i + 1

    return theil / array_size

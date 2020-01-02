cimport cython
from libc.math cimport log, fabs, sqrt

cdef extern from "math.h":
    float INFINITY


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float var_agg(int cnt, float sum1, float sum2):
    """Calculate variance of a given array by using equation : variance = sums of squares - square of sums
        Parameters 
        ----------
            cnt : int, number of array elements
            sum1 : float, sum of array elements
            sum2 : float, sum of squared array elements
        Returns 
        ----------
            result : float, variance of array
    """

    cdef float result = (sum2/(cnt + 0.01)) - (sum1/(cnt + 0.01))**2
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float sum_squared_array(float [:] y, int n):
    """Sum squared elements of a given array
        Parameters
        ----------
            y : array of type np.float32, input array
            n : np.int32, length of array
        Returns 
        ----------
            result : np.float32, sum of squared array elements
    """

    cdef int i = 0
    cdef float result = 0.0
    while i < n:
        result += y[i] ** 2
        i += 1

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float sum_array(float [:] y, int n):
    """Sum elements of a given array
        Parameters
        ----------
            y : array of type np.float32, input array
            n : np.int32, length of array
        Returns 
        ----------
            result : np.float32, sum of array elements
    """

    cdef int i = 0
    cdef float result = 0.0
    while i < n:
        result += y[i]
        i += 1

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float weighted_variance(int split_index, float [:] y, int len_y):
    """Calculate weighted of given split array after splitting at given index
        Parameters
        ----------
            split_index : np.int32, index of array element to perform split
            y : array of type np.float32, input array
            len_y : np.int32, length of array
        Returns
        ----------
            result : Theil index of given split array after splitting at given index
    """
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
    """Calculate variance of given array.
        Parameters 
        ----------
            y : array of type np.float32, input array
            array_size : np.int32, length of array
        Returns
        ----------
            result : variance
    """

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
                y : numpy array of type np.float32, array of objects' labels, order by objects' similarity
                s : numpy array of type np.float32, array of similarities
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
    """This is a function calculating optimal split according to criterion of minimizing weighted variance
            Parameters
            ---------
                y : numpy array of type np.float32, array of objects' labels, order by objects' similarity
                s : numpy array of type np.float32, array of similarities
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
    """Find split point minimizing weighted variance
            Parameters
            ---------
                y : numpy array of type np.float32, array of objects' labels, order by objects' similarity
                s : numpy array of type np.float32, array of similarities
                max_range : np.int32, should be length of y array - 1, so at least one point is left in right partition
            Returns
            ---------
                best_split_idx : np.int32, index of element at which optimal split should be performed
    """

    cdef float best_impurity = INFINITY
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
    """This is a function calculating optimal split according to criterion of minimizing Theil index
            Parameters
            ---------
                y : numpy array of type np.float32, array of objects' labels, order by objects' similarity
                s : numpy array of type np.float32, array of similarities
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

    assert best_split_idx >= 0, 'split index should be >= 0'
    return best_split_idx, best_impurity


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float theil_index(int split_index, float [:] y, int len_y):
    """Calculate Theil index of given split array after splitting at given index
        Parameters
        ----------
            split_index : np.int32, index of array element to perform split
            y : array of type np.float32, input array
            len_y : np.int32, length of array
        Returns
        ----------
            result : Theil index of given split array after splitting at given index
    """

    cdef float [:] left_partition = y[:split_index]
    cdef float [:] right_partition = y[split_index:]
    cdef int len_left_partition = left_partition.shape[0]
    cdef int len_right_partition = len_y - len_left_partition

    cdef float flen_left_partition = left_partition.shape[0]
    cdef float left_proportion = flen_left_partition / len_y

    cdef float result = left_proportion * theil(left_partition, len_left_partition) + \
           (1.0 - left_proportion) * theil(right_partition, len_right_partition)

    assert result >= 0.0, 'result should be >= 0.0'
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float theil(float [:] y, int array_size):
    """Calculate Theil index of given array.
        Parameters 
        ----------
            y : array of type np.float32, input array
            array_size : np.int32, length of array
        Returns
        ----------
            result : Theil index
    """

    assert array_size > 0, 'array of size 0'
    if array_size == 1:
        return 0.0

    cdef float array_sum = 0.0
    cdef int i = 0
    # sprawdzić czy w obecnej partycji wszystkie y są takie same
    while i < array_size:
        array_sum = array_sum + y[i]
        i = i + 1

    cdef float mean = array_sum / array_size + 0.001


    cdef float theil = 0.0
    cdef float t = 0.0
    i = 0
    while i < array_size:
        t = y[i] / mean
        if t == 0.0:
            i = i + 1
            continue
        theil = theil + t * log(t)
        i = i + 1

    cdef float result = theil / array_size
    assert result >= -0.1, 'Negative Theil index'
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def find_split_atkinson(float [:] y, float [:] s, int max_range):
    """This is a function calculating optimal split according to criterion of minimizing Atkinson index
            Parameters
            ---------
                y : numpy array of type np.float32, array of objects' labels, order by objects' similarity
                s : numpy array of type np.float32, array of similarities
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

        if s[i] == s[i+1]:
            i += 1
            continue

        curr_impurity = atkinson_index(i+1, y, len_y)

        if curr_impurity < best_impurity:
            best_impurity = curr_impurity
            best_split_idx = i

        i += 1

    assert best_split_idx >= 0, 'split index should be >= 0'
    return best_split_idx, best_impurity


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float atkinson_index(int split_index, float [:] y, int len_y):
    """Calculate Atkinson index of given split array after splitting at given index
        Parameters
        ----------
            split_index : np.int32, index of array element to perform split
            y : array of type np.float32, input array
            len_y : np.int32, length of array
        Returns
        ----------
            result : Atkinson index of given split array after splitting at given index
    """

    cdef float [:] left_partition = y[:split_index]
    cdef float [:] right_partition = y[split_index:]
    cdef int len_left_partition = left_partition.shape[0]
    cdef int len_right_partition = len_y - len_left_partition

    cdef float flen_left_partition = left_partition.shape[0]
    cdef float left_proportion = flen_left_partition / len_y

    cdef float result = left_proportion * atkinson(left_partition, len_left_partition) + \
           (1.0 - left_proportion) * atkinson(right_partition, len_right_partition)

    assert result >= -0.1, 'Negative Atkinson index'
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float atkinson(float [:] y, int array_size):
    """Calculate Atkinson index of given array.
        Parameters 
        ----------
            y : array of type np.float32, input array
            array_size : np.int32, length of array
        Returns
        ----------
            result : Atkinson index
    """

    assert array_size > 0, 'array of size 0'
    if array_size == 1:
        return 0.0

    cdef float array_sum = 0.0
    cdef float array_sqrt_sum = 0.0
    cdef int i = 0

    while i < array_size:
        array_sum = array_sum + y[i]
        array_sqrt_sum = array_sqrt_sum + sqrt(y[i])
        i = i + 1

    return 1 - array_sqrt_sum ** 2 / (array_sum * array_size)

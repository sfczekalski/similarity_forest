cimport cython
from libc.math cimport exp

import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[dtype=np.float32_t, ndim=1] dot_projection(float [:, :] X, float [:] p, float [:] q):
    """Projection of data-point on split direction using dot product.
        Parameters
        ----------
            X : memoryview of ndarray, data-points to be projected
            p : memoryview of ndarray, first data-point used to draw split direction
            q : memoryview of ndarray, second data-point used to draw split direction
        Returns 
        ----------
            result : float value
    """
    cdef float s = 0.0
    cdef float d = 0.0
    cdef int n = X.shape[0]
    cdef int m = X.shape[1]
    cdef int i = 0
    cdef int j = 0

    cdef np.ndarray[dtype=np.float32_t, ndim=1] result = np.zeros(shape=n, dtype=np.float32)

    with nogil:
        for i in range(n):
            s = 0.0
            for j in range(m):
                d = q[j] - p[j]
                s += X[i, j] * d

            result[i] = s

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float sqeuclidean_projection(float [:] xi, float [:] p, float [:] q) nogil:
    """Projection of data-point on split direction using squared euclidean distance.
        It serves as an approximation of euclidean distance, when sorted using both methods, 
        the order of data-points remains the same. 
        Parameters
        ----------
            xi : memoryview of ndarray, data-point to be projected
            p : memoryview of ndarray, first data-point used to draw split direction
            q : memoryview of ndarray, second data-point used to draw split direction
        Returns 
        ----------
            result : float value
    """
    cdef float result = 0.0
    cdef int n = xi.shape[0]
    cdef int i = 0
    cdef float p_q

    for i in range(n):
        p_q = p[i] - q[i]
        result += xi[i] * p_q

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float rbf_projection(float [:] xi, float [:] p, float [:] q) nogil:
    """Projection of data-point on split direction using squared euclidean distance.
        It serves as an approximation of euclidean distance, when sorted using both methods, 
        the order of data-points remains the same. 
        Parameters
        ----------
            xi : memoryview of ndarray, data-point to be projected
            p : memoryview of ndarray, first data-point used to draw split direction
            q : memoryview of ndarray, second data-point used to draw split direction
        Returns 
        ----------
            result : float value
    """
    cdef float result = 0.0
    cdef float xq = 0.0
    cdef float xp = 0.0
    cdef int n = xi.shape[0]
    cdef float gamma = 1 / <float>len(xi)
    cdef int i = 0
    cdef float temp_x_q
    cdef float temp_x_p

    for i in range(n):
        temp_x_q = xi[i] - q[i]
        xq += temp_x_q ** temp_x_q

        temp_x_p = xi[i] - p[i]
        xp += temp_x_p ** temp_x_p

    xq = exp(-gamma * xq)
    xp = exp(-gamma * xp)

    result = xq - xp

    return result
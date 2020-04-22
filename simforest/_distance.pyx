import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
import numexpr as ne


cpdef float rbf_sequential(float [:] u, float [:] v):
    cdef float gamma = 1.0 / len(u)

    cdef float sqeuclidean_dist = sqeuclidean_sequential(u, v)
    cdef float result = exp(-gamma * sqeuclidean_dist)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float sqeuclidean_sequential(float [:] u, float [:] v) nogil:
    """Calcuate squared euclidean distance of two vectors. 
        It serves as an approximation of euclidean distance, when sorted using both methods, 
        the order of data-points remains the same. 
        Parameters
        ----------
            u : memoryview of ndarray, first vector
            v : memoryview of ndarray, second vector
        Returns 
        ----------
            result : float value
    """
    cdef float result = 0.0
    cdef int n = u.shape[0]
    cdef int i = 0
    for i in range(n):
        result += (u[i] - v[i]) ** 2

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float dot_sequential(float [:] u, float [:] v) nogil:
    """Calcuate dot product of two vectors.
        Parameters
        ----------
            u : memoryview of ndarray, first vector
            v : memoryview of ndarray, second vector
        Returns 
        ----------
            result : float value
    """
    cdef float result = 0.0
    cdef int n = u.shape[0]
    cdef int i = 0
    for i in range(n):
        result += u[i] * v[i]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float dot_projection_sequential(float [:] xi, float [:] p, float [:] q) nogil:
    """Projection of data-point on split direction using dot product.
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

    for i in range(n):
        result += xi[i] * q[i] - xi[i] * p[i]

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float sqeuclidean_projection_sequential(float [:] xi, float [:] p, float [:] q) nogil:
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

    for i in range(n):
        result += (xi[i] - q[i]) ** 2 - (xi[i] - p[i]) ** 2

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float rbf_projection_sequential(float [:] xi, float [:] p, float [:] q) nogil:
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
    cdef int n = len(xi)
    cdef float gamma = 1 / <float>len(xi)
    cdef int i = 0

    for i in range(n):
        xq += (xi[i] - q[i]) ** 2

    xq = exp(-gamma * xq)

    for i in range(n):
        xp += (xi[i] - p[i]) ** 2

    xp = exp(-gamma * xp)

    result = xq - xp

    return result


cdef np.ndarray rbf(np.ndarray X, np.ndarray p, np.ndarray q):
    """A function calculating rbf kernel based projection of data-points in matrix X
            X : array of shape=(n_examples, n_features),
                should be 2-dimensional, even if it consists only of one data-point!
            p : array, first data-point used for projection
            q : array, second data-point used for projection
            gamma: float, default None, gamma in rbf computation
                If None, defaults to 1 / num_features
        Returns
        ----------
            projection : array of shape=(n_examples,) with distance-based projection values

    """
    cdef float gamma = 1.0 / X.shape[1]

    cdef np.ndarray result = np.exp(-gamma * (ne.evaluate("(X - q) ** 2").sum(1))) -\
                             np.exp(-gamma * (ne.evaluate("(X - p) ** 2").sum(1)))
    return result


cdef np.ndarray sqeuclidean(np.ndarray X, np.ndarray p, np.ndarray q):
    """A function calculating squared euclidean distance based projection of data-points in matrix X
        Parameters
        ----------
            X : array of shape=(n_examples, n_features),
                should be 2-dimensional, even if it consists only of one data-point!
            p : array, first data-point used for projection
            q : array, second data-point used for projection

        Returns
        ----------
            projection : array of shape=(n_examples,) with distance-based projection values

        Notes
        ----------
            the other option is : np.linalg.norm(X - q, axis=1) - np.linalg.norm(X - p, axis=1)
    """

    cdef np.ndarray result = ne.evaluate("(X - q) ** 2").sum(1) - ne.evaluate("(X - p) ** 2").sum(1)
    return result

cdef np.ndarray euclidean(np.ndarray X, np.ndarray p, np.ndarray q):
    """A function calculating euclidean distance based projection of data-points in matrix X
        Parameters
        ----------
            X : array of shape=(n_examples, n_features),
                should be 2-dimensional, even if it consists only of one data-point!
            p : array, first data-point used for projection
            q : array, second data-point used for projection

        Returns
        ----------
            projection : array of shape=(n_examples,) with distance-based projection values

        Notes
        ----------
            the other option is : np.linalg.norm(X - q, axis=1) - np.linalg.norm(X - p, axis=1)
    """

    cdef np.ndarray result =  np.sqrt(ne.evaluate("(X - q) ** 2").sum(1)) - np.sqrt(ne.evaluate("(X - p) ** 2").sum(1))
    return result

cdef np.ndarray dot_product(np.ndarray X, np.ndarray p, np.ndarray q):
    """A function calculating dot product distance based projection of data-points in matrix X
        Parameters
        ----------
            X : array of shape=(n_examples, n_features),
                should be 2-dimensional, even if it consists only of one data-point!
            p : array, first data-point used for projection
            q : array, second data-point used for projection

        Returns
        ----------
            projection : array of shape=(n_examples,) with distance-based projection values

        Notes
        ----------
            the other option is : sgemm(alpha=1.0, a=X, b=X, trans_b=True)
    """

    cdef np.ndarray result = np.dot(X, q) - np.dot(X, p)
    return result

cpdef float sigmoid_projection_sequential(float [:] x, float [:] p, float [:] q) nogil:

    cdef float result = 0.0
    cdef float xq = 0.0
    cdef float xp = 0.0
    cdef int n = x.shape[0]
    cdef float gamma = 1 / <float>len(x)
    cdef int i = 0

    for i in range(n):
        xq += x[i] * q[i]

    xq = 1 / (1 + exp(-gamma * xq))

    for i in range(n):
        xp += x[i] * p[i]

    xp = 1 / (1 + exp(-gamma * xp))

    result = xq - xp

    return result

cdef np.ndarray sigmoid(float [:, :] X, float [:] p, float [:] q):
    """A function calculating sigmoid distance based projection of data-points in matrix X
        Parameters
        ----------
            X : array of shape=(n_examples, n_features),
                should be 2-dimensional, even if it consists only of one data-point!
            p : array, first data-point used for projection
            q : array, second data-point used for projection

        Returns
        ----------
            projection : array of shape=(n_examples,) with distance-based projection values

    """
    cdef float alpha = 1 / <float>len(p)
    cdef float result = 1 / (1 + np.exp(-alpha * np.dot(X, q))) - 1 / (1 + np.exp(-alpha * np.dot(X, p)))
    return result
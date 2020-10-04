# from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
# https://cython.readthedocs.io/en/latest/src/tutorial/memory_allocation.html
import numpy as np
cimport numpy as np
cimport cython
from scipy.special import comb
from sklearn.utils.validation import check_random_state
from cython.parallel import prange, parallel
cimport openmp
from libc.math cimport exp

from projection cimport dot_projection


# projection function type definition
ctypedef np.ndarray[dtype=np.float32_t, ndim=1] (*f_type)(float [:, :] X, float [:] p, float [:] q)
"""This type represents a function type for a function that calculates projection of data-points on split direction.
    Parameters
    ----------
        X : memoryview of ndarray, data-points to be projected
        p : memoryview of ndarray, first data-point used for drawing split direction
        q : memoryview of ndarray, second data-point used for drawing split direction
    Returns 
    ----------
        float, value of projection on given split direction
"""


cdef class CSimilarityForestClusterer:
    """Similarity forest clusterer."""

    cdef random_state
    cdef str sim_function
    cdef int max_depth
    cdef public list estimators_
    cdef int n_estimators
    cdef int bootstrap

    def __cinit__(self,
                  random_state=None,
                  str sim_function='dot',
                  int max_depth=5,
                  int n_estimators = 20,
                  int bootstrap = 0):
        self.random_state = random_state
        self.sim_function = sim_function
        self.max_depth = max_depth
        self.estimators_ = []
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef CSimilarityForestClusterer fit(self, np.ndarray[np.float32_t, ndim=2] X):
        """Build a forest of trees from the training set X.
            Parameters
            ----------
            X : array-like matrix of shape = [n_samples, n_features]
                The training data samples.
            Returns
            -------
            self : CSimilarityForestClusterer
        """
        cdef int n = X.shape[0]
        cdef dict args = dict()

        cdef random_state = check_random_state(self.random_state)
        if self.random_state is not None:
            args['random_state'] = self.random_state

        if self.max_depth != -1:
            args['max_depth'] = self.max_depth

        args['sim_function'] = self.sim_function

        cdef int [:] indicies
        for i in range(self.n_estimators):
            if self.bootstrap == 0:
                self.estimators_.append(CSimilarityTreeCluster(**args).fit(X))
            else:
                indicies = random_state.choice(range(n), n, replace=True).astype(np.int32)
                self.estimators_.append(CSimilarityTreeCluster(**args).fit(X[indicies]))


        return self

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float32_t, ndim=1] predict_(self, np.ndarray[np.float32_t, ndim=2] X):
        """Produce pairwise distance matrix.
            Parameters
            ----------
            X : array-like matrix of shape = [n_samples, n_features]
                The training data samples.
            Returns
            -------
            distance_matrix : ndarray of shape = comb(n_samples, 2) containing the distances
        """
        cdef int n = X.shape[0]
        cdef float [:, :] X_i_view
        cdef float [:, :] X_j_view

        cdef np.ndarray[np.float32_t, ndim=1] distance_matrix = np.ones(<int>comb(n, 2), np.float32, order='c')
        cdef float [:] view = distance_matrix

        cdef int diagonal = 1
        cdef int idx = 0
        cdef float similarity = 0.0
        cdef int i = 0
        cdef int j = 0
        cdef int e = 0
        cdef CSimilarityTreeCluster current_tree

        for i in range(n):
            for j in range(diagonal, n):
                for e in range(self.n_estimators,):
                    current_tree = self.estimators_[e]
                    X_i_view = X[np.newaxis, i]
                    X_j_view = X[np.newaxis, j]
                    similarity += current_tree.similarity(X_i_view, X_j_view)

                # similarity is an average depth at which points split across all trees
                # but we don't need to divide the accumulated similarities as this only rescales the distance
                view[idx] = 1 / <float>similarity
                similarity = 0.0
                idx += 1
            diagonal += 1

        return distance_matrix


cdef class CSimilarityTreeCluster:
    """Similarity Tree clusterer."""

    cdef random_state
    cdef str sim_function
    cdef int max_depth
    cdef int depth
    cdef int is_leaf
    cdef float [:] _p
    cdef float [:] _q
    cdef float _split_point
    cdef int [:] lhs_idxs
    cdef int [:] rhs_idxs
    cdef CSimilarityTreeCluster _lhs
    cdef CSimilarityTreeCluster _rhs
    cdef _rng
    cdef f_type projection

    def __cinit__(self,
                  random_state=None,
                  str sim_function='dot',
                  int max_depth=-1,
                  int depth=1):
        self.random_state = random_state
        self.sim_function= sim_function
        self.max_depth = max_depth
        self.depth = depth
        self.is_leaf = 0


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int is_pure(self, float [:, :] X) nogil:
        """Check if all data-points in the matrix are the same.
            Parameters
            ----------
            X : memoryview of ndarray of shape = [n_samples, n_features]
                The data-points.
            Returns
            -------
            pure : int, 0 indicates that the array is not pure, 1 that it is
        
        """
        cdef int n = X.shape[0]
        cdef int m = X.shape[1]
        cdef int pure = 1
        cdef int i = 0
        cdef int j = 0

        for i in range(n-1):
            for j in range(m):
                # found different row! Not pure
                if X[i, j] != X[i+1, j]:
                    pure = 0
                    break

        return pure

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int sample_split_direction(self, np.ndarray[np.float32_t, ndim=2] X, int first):
        """Sample index of second data-point to draw split direction. 
            First one was sampled in fit, here we sample only the second one in order to avoid passing a tuple as a result
            Parameters
            ----------
            X : memoryview of ndarray of shape = [n_samples, n_features]
                The data-points.
            first : int, index of first data-point
            Returns
            -------
            second : int, index of second data-point
        
        """
        cdef int n = X.shape[0]
        cdef int m = X.shape[1]
        cdef float [:] first_row = X[first]

        cdef np.ndarray[np.int32_t, ndim=1] others = np.where(np.abs(X - X[first]) > 0)[0].astype(np.int32)
        #assert len(others) > 0, 'All points are the same'
        return self._rng.choice(others, replace=False)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _find_split(self, float [:, :] X, f_type projection):
        """Project all data-points, find a split point and indexes of both child partitions.
            Parameters
            ----------
            X : memoryview of ndarray of shape = [n_samples, n_features]
                The data-points.
            projection : f_type, a function used to project data-points
            Returns
            -------
            void
        
        """

        # Calculate similarities
        cdef int n = X.shape[0]
        cdef np.ndarray[np.float32_t, ndim=1] similarities = np.empty(n, dtype=np.float32, order='c')
        cdef float [:] p = self._p
        cdef float [:] q = self._q

        similarities = projection(X, p, q)

        cdef float similarities_min = np.min(similarities)
        cdef float similarities_max = np.max(similarities)

        # Find random split point
        self._split_point = self._rng.uniform(similarities_min, similarities_max, 1)

        # Find indexes of points going left
        self.lhs_idxs = np.nonzero(similarities <= self._split_point)[0].astype(np.int32)
        self.rhs_idxs = np.invert(self.lhs_idxs)


    cpdef CSimilarityTreeCluster fit(self, np.ndarray[np.float32_t, ndim=2] X):
        """Build a tree from the training set X.
            Parameters
            ----------
            X : array-like matrix of shape = [n_samples, n_features]
                The training data samples.
            Returns
            -------
            self : CSimilarityTreeCluster
        """
        cdef int n = X.shape[0]
        if n <= 1:
            self.is_leaf = 1
            return self

        if self.is_pure(X) == 1:
            self.is_leaf = 1
            return self

        if self.max_depth > -1:
            if self.depth == self.max_depth:
                self.is_leaf = 1
                return self

        self._rng = check_random_state(self.random_state)

        cdef int p = 0
        cdef int q = 1
        # if more that two points, find a split
        if n > 2:
            # sample p randomly
            p = self._rng.randint(0, n)

            # sample q so that it's not a copy the same point
            q = self.sample_split_direction(X, p)
            if q == -1:
                raise ValueError(f'Could not find second split point; is_pure should handle that! {np.asarray(X)}')

        self._p = X[p]
        self._q = X[q]

        if self.sim_function == 'dot':
            self.projection = dot_projection
        else:
            raise ValueError('Unknown similarity function')

        self._find_split(X, self.projection)

        # if split has been found
        if X[self.lhs_idxs].shape[0] > 0 and X[self.rhs_idxs].shape[0] > 0:
            self._lhs = CSimilarityTreeCluster(random_state=self.random_state,
                                               sim_function=self.sim_function,
                                               max_depth=self.max_depth,
                                               depth=self.depth+1).fit(X[self.lhs_idxs])

            self._rhs = CSimilarityTreeCluster(random_state=self.random_state,
                                               sim_function=self.sim_function,
                                               max_depth=self.max_depth,
                                               depth=self.depth+1).fit(X[self.rhs_idxs])
        else:
            self.is_leaf = 1
            return self

        return self

    cdef int similarity(self, float [:, :] xi, float [:, :] xj):
        """Calculate similarity of a pair of data-points in tree-space.
            The pair of traverses down the tree, and the depth on which the pair splits is recorded.
            This values serves as a similarity measure between the pair.
            Parameters
            ----------
            X : array-like matrix of shape = [n_samples, n_features]
                The training data samples.
            Returns
            -------
            int : the depth on which the pair splits.
        """
        if self.is_leaf:
            return self.depth

        cdef bint path_i = self.projection(xi, self._p, self._q) <= self._split_point
        cdef bint path_j = self.projection(xj, self._p, self._q) <= self._split_point

        if path_i == path_j:
            # the same path, check if the pair goes left or right
            if path_i:
                return self._lhs.similarity(xi, xj)
            else:
                return self._rhs.similarity(xi, xj)
        else:
            # different path, return current depth
            return self.depth


    cpdef np.ndarray[np.float32_t, ndim=1] predict_(self, np.ndarray[np.float32_t, ndim=2] X):
        """Produce pairwise distance matrix according to single tree.
            Parameters
            ----------
            X : array-like matrix of shape = [n_samples, n_features]
                The training data samples.
            Returns
            -------
            distance_matrix : ndarray of shape = comb(n_samples, 2) containing the distances
        """
        cdef int n = X.shape[0]
        cdef np.ndarray[np.float32_t, ndim=1] distance_matrix = np.ones(<int>comb(n, 2), dtype=np.float32, order='c')
        cdef float [:] view = distance_matrix

        cdef int diagonal = 1
        cdef int idx = 0
        for c in range(n):
            for r in range(diagonal, n):
                view[idx] = 1 / <float>self.similarity(X[c], X[r])
                idx += 1
            diagonal += 1

        return distance_matrix

from libc.stdlib cimport srand, rand, RAND_MAX, malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
# https://cython.readthedocs.io/en/latest/src/tutorial/memory_allocation.html
import numpy as np
cimport numpy as np
cimport cython
from scipy.special import comb
from sklearn.utils.validation import check_random_state
cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)


cdef class CSimilarityForestClusterer:

    cdef random_state
    cdef str sim_function
    cdef int max_depth
    cdef list estimators_
    cdef int n_estimators

    def __cinit__(self,
                  random_state=None,
                  str sim_function='euclidean',
                  int max_depth=-1,
                  int n_estimators = 20):
        self.random_state = random_state
        self.sim_function= sim_function
        self.max_depth = max_depth
        self.estimators_ = []
        self.n_estimators = n_estimators

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fit(self, float [:, :] X):

        cdef int n = X.shape[0]
        cdef dict args = dict()

        cdef random_state = check_random_state(self.random_state)
        if self.random_state is not None:
            args['random_state'] = self.random_state

        if self.max_depth is not None:
            args['max_depth'] = self.max_depth

        cdef int [:] indicies
        for i in range(self.n_estimators):
            indicies = random_state.choice(range(n), n, replace=True).astype(np.int32)
            self.estimators_.append(CSimilarityTreeCluster(**args).fit(X.base[indicies]))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray predict(self, float [:, :] X):
        cdef int n = X.shape[0]
        cdef np.ndarray dinstance_matrix = np.ones(<int>comb(n, 2), dtype=np.float32)
        cdef float [:] view = dinstance_matrix


        cdef int diagonal = 1
        cdef int idx = 0
        cdef float dist = 0.0
        for i in range(n):
            for j in range(diagonal, n):
                for e in range(self.n_estimators):
                    dist += self.estimators_[e].distance(X.base[i], X.base[j])

                dist = dist/<float>self.n_estimators
                view[idx] = 1 / <float>dist
                idx += 1
            diagonal += 1

        return dinstance_matrix


cdef class CSimilarityTreeCluster:

    cdef int random_state
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

    def __cinit__(self,
                  int random_state=-1,
                  str sim_function='euclidean',
                  int max_depth=-1,
                  int depth=1):
        self.random_state = random_state
        self.sim_function= sim_function
        self.max_depth = max_depth
        self.depth = depth
        self.is_leaf = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int is_pure(self, float [:, :] X):
        cdef int i = 0
        cdef int j = 0
        cdef int n = X.shape[0]
        cdef int m = X.shape[1]
        cdef int pure = 1

        while i < n-1:
            while j < m:
                # found different raw! Not pure
                if X[i, j] != X[i+1, j]:
                    pure = 0
                    break
                j += 1
            j = 0
            i += 1

        return pure

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int sample_split_direction(self, float [:, :] X, int first):
        cdef int n = X.shape[0]
        cdef int m = X.shape[1]
        cdef float [:] first_row = X[first]
        cdef int i = 0
        cdef int j = 0

        while i < n:
            # sample random row
            # iterate over its elements, if at least one differs - then its different than first_row
            while j < m:
                if X[i, j] != first_row[j]:
                    return i
                j += 1
            j = 0
            i += 1

        # q not found
        return -1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float sqeuclidean(self, float [:] u, float [:] v):
        cdef float result = 0.0
        cdef int i = 0
        cdef int n = u.shape[0]
        while i < n:
            result += (u[i] - v[i]) ** 2
            i += 1

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _find_split(self, float [:, :] X):

        # Calculate similarities
        cdef int n = X.shape[0]
        cdef np.ndarray array = np.zeros(n, dtype=np.float32)
        cdef float [:] similarities = array

        similarities[0] = self.sqeuclidean(X[0], self._q) - self.sqeuclidean(X[0], self._p)
        cdef float similarities_min = similarities[0]
        cdef float similarities_max = similarities[0]

        cdef int i = 1
        while i < n:
            similarities[i] = self.sqeuclidean(X[i], self._q) - self.sqeuclidean(X[i], self._p)

            if similarities[i] < similarities_min:
                similarities_min = similarities[i]

            if similarities[i] > similarities_max:
                similarities_max = similarities[i]

            i += 1

        # Find random split point
        if self.random_state > -1:
            srand(self.random_state)
        else:
            srand(time(NULL))
        self._split_point = similarities_min + (similarities_max - similarities_min) * rand()/<float>RAND_MAX

        # Find indexes of points going left
        self.lhs_idxs = np.nonzero(array <= self._split_point)[0].astype(np.int32)
        self.rhs_idxs = np.nonzero(array > self._split_point)[0].astype(np.int32)

    cpdef fit(self, float [:, :] X):
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

        cdef int p = 0
        cdef int q = 1
        # if more that two point, find a split
        if n > 2:
            # sample p randomly
            if self.random_state > -1:
                srand(self.random_state)
            else:
                srand(time(NULL))
            p = <int>(rand()/<float>RAND_MAX * n)

            # sample q so that it's not a copy the same point
            q = self.sample_split_direction(X, p)
            if q == -1:
                raise ValueError(f'Could not find second split point; is_pure should handle that! {np.asarray(X)}')

        self._p = X[p]
        self._q = X[q]

        self._find_split(X)

        self._lhs = CSimilarityTreeCluster(random_state=self.random_state,
                                           sim_function=self.sim_function,
                                           max_depth=self.max_depth,
                                           depth=self.depth+1).fit(X.base[self.lhs_idxs])

        self._rhs = CSimilarityTreeCluster(random_state=self.random_state,
                                           sim_function=self.sim_function,
                                           max_depth=self.max_depth,
                                           depth=self.depth+1).fit(X.base[self.rhs_idxs])

        return self

    cpdef int distance(self, float [:] xi, float [:] xj):
        if self.is_leaf:
            return self.depth

        cdef bint path_i = self.sqeuclidean(xi, self._q) - self.sqeuclidean(xi, self._p) <= self._split_point
        cdef bint path_j = self.sqeuclidean(xj, self._q) - self.sqeuclidean(xj, self._p) <= self._split_point


        if path_i == path_j:
            # the same path, check if go left or right
            if path_i:
                return self._lhs.distance(xi, xj)
            else:
                return self._rhs.distance(xi, xj)
        else:
            # different path, return current depth
            return self.depth


    cpdef np.ndarray predict(self, float [:, :] X):
        cdef int n = X.shape[0]
        cdef np.ndarray dinstance_matrix = np.ones(<int>comb(n, 2), dtype=np.float32)
        cdef float [:] view = dinstance_matrix


        cdef int diagonal = 1
        cdef int idx = 0
        for c in range(n):
            for r in range(diagonal, n):
                view[idx] = 1 / <float>self.distance(X.base[c], X.base[r])
                idx += 1
            diagonal += 1

        return dinstance_matrix
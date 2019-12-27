from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
# https://cython.readthedocs.io/en/latest/src/tutorial/memory_allocation.html
import numpy as np
cimport numpy as np
cimport cython
from scipy.special import comb
from sklearn.utils.validation import check_random_state

cdef class CSimilarityForestClusterer:

    cdef random_state
    cdef str sim_function
    cdef int max_depth
    cdef public list estimators_
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
    cpdef CSimilarityForestClusterer fit(self, np.ndarray[np.float32_t, ndim=2] X):
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
            self.estimators_.append(CSimilarityTreeCluster(**args).fit(X[indicies]))

        return self

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float32_t, ndim=1] predict_(self, np.ndarray[np.float32_t, ndim=2] X):
        cdef int n = X.shape[0]
        cdef np.ndarray[np.float32_t, ndim=1] dinstance_matrix = np.ones(<int>comb(n, 2), np.float32)
        cdef float [:] view = dinstance_matrix


        cdef int diagonal = 1
        cdef int idx = 0
        cdef float dist = 0.0
        for i in range(n):
            for j in range(diagonal, n):
                for e in range(self.n_estimators):
                    dist += self.estimators_[e].distance(X[i], X[j])

                dist = dist/<float>self.n_estimators
                view[idx] = 1 / <float>dist
                idx += 1
            diagonal += 1

        return dinstance_matrix


cdef class CSimilarityTreeCluster:

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

    def __cinit__(self,
                  random_state=None,
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
        cdef int n = X.shape[0]
        cdef int m = X.shape[1]
        cdef int pure = 1

        for i in range(n-1):
            for j in range(m):
                # found different raw! Not pure
                if X[i, j] != X[i+1, j]:
                    pure = 0
                    break

        return pure

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int sample_split_direction(self, np.ndarray[np.float32_t, ndim=2] X, int first):
        cdef int n = X.shape[0]
        cdef int m = X.shape[1]
        cdef float [:] first_row = X[first]

        cdef np.ndarray[np.int32_t, ndim=1] others = np.where(np.abs(X - X[first]) > 0)[0].astype(np.int32)
        #assert len(others) > 0, 'All points are the same'
        return self._rng.choice(others, replace=False)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float sqeuclidean(self, float [:] u, float [:] v) nogil:
        cdef float result = 0.0
        cdef int n = u.shape[0]
        for i in range(n):
            result += (u[i] - v[i]) ** 2

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _find_split(self, np.ndarray[np.float32_t, ndim=2] X):

        # Calculate similarities
        cdef int n = X.shape[0]
        cdef np.ndarray[np.float32_t, ndim=1] array = np.zeros(n, dtype=np.float32)
        cdef float [:] similarities = array

        similarities[0] = self.sqeuclidean(X[0], self._q) - self.sqeuclidean(X[0], self._p)
        cdef float similarities_min = similarities[0]
        cdef float similarities_max = similarities[0]

        for i in range(1, n):
            similarities[i] = self.sqeuclidean(X[i], self._q) - self.sqeuclidean(X[i], self._p)

            if similarities[i] < similarities_min:
                similarities_min = similarities[i]

            if similarities[i] > similarities_max:
                similarities_max = similarities[i]


        # Find random split point
        self._split_point = self._rng.uniform(similarities_min, similarities_max, 1)

        # Find indexes of points going left
        self.lhs_idxs = np.nonzero(array <= self._split_point)[0].astype(np.int32)
        self.rhs_idxs = np.nonzero(array > self._split_point)[0].astype(np.int32)

    cpdef CSimilarityTreeCluster fit(self, np.ndarray[np.float32_t, ndim=2] X):
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

        self._find_split(X)

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
            raise ValueError('Error when finding a split: all points go to the same partition')

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


    cpdef np.ndarray[np.float32_t, ndim=1] predict_(self, np.ndarray[np.float32_t, ndim=2] X):
        cdef int n = X.shape[0]
        cdef np.ndarray[np.float32_t, ndim=1] distance_matrix = np.ones(<int>comb(n, 2), dtype=np.float32)
        cdef float [:] view = distance_matrix


        cdef int diagonal = 1
        cdef int idx = 0
        for c in range(n):
            for r in range(diagonal, n):
                view[idx] = 1 / <float>self.distance(X[c], X[r])
                idx += 1
            diagonal += 1

        return distance_matrix

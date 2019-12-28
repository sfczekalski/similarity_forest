from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
# https://cython.readthedocs.io/en/latest/src/tutorial/memory_allocation.html
import numpy as np
cimport numpy as np
cimport cython
from scipy.special import comb
from sklearn.utils.validation import check_random_state
from cython.parallel import prange, parallel
cimport openmp

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
        cdef np.ndarray[np.float32_t, ndim=1] dinstance_matrix = np.ones(<int>comb(n, 2), np.float32, order='c')
        cdef float [:] view = dinstance_matrix

        cdef int num_threads = 4
        cdef int diagonal = 1
        cdef int idx = 0
        cdef float similarity = 0.0
        cdef int i = 0
        cdef int j = 0
        cdef int e = 0
        for i in range(n):
            for j in range(diagonal, n):
                for e in range(self.n_estimators,):
                    similarity += self.estimators_[e].distance(X[i], X[j])

                # similarity is an average depth at which points split across all trees
                #similarity = similarity/<float>self.n_estimators
                # distance = 1 / similarity
                view[idx] = 1 / <float>similarity
                similarity = 0.0
                idx += 1
            diagonal += 1

        return dinstance_matrix

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float32_t, ndim=2] ppredict_(self, np.ndarray[np.float32_t, ndim=2] X):
        """
        Notes
        ------
            In parallel implementation distance is calculated as a sum of 1/similarity across the trees,
            instead of 1 / sum of similarities.
            
            Parallel implementation materializes the whole N*N distance matrix instead of comb(N, 2) flat array.
            Possibly change it in the future.
            
        """
        cdef int n = X.shape[0]
        cdef float [:, :] X_view = X

        cdef np.ndarray[np.float32_t, ndim=2] dinstance_matrix = np.zeros(shape=(n, n), dtype=np.float32)
        cdef float [:, :] dinstance_matrix_view = dinstance_matrix

        cdef float similarity = 0.0

        cdef int num_threads = 8
        cdef int diagonal = 1
        cdef int i = 0
        cdef int j = 0
        cdef int e = 0
        cdef CSimilarityTreeCluster current_tree

        for e in range(self.n_estimators):
            current_tree = self.estimators_[e]
            for i in range(n):
                for j in prange(n, nogil=True, schedule='dynamic', num_threads=num_threads):
                    if i == j:
                        continue
                    similarity = current_tree.distance(X_view[i], X_view[j])
                    dinstance_matrix_view[i, j] += 1 / <float>similarity
                    dinstance_matrix_view[j, i] = dinstance_matrix_view[i, j]


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
        cdef int i =0
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
        cdef int i = 0
        for i in range(n):
            result += (u[i] - v[i]) ** 2

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float sqeuclidean_projection(self, float [:] xi) nogil:
        cdef float result = 0.0
        cdef int n = xi.shape[0]
        cdef int i = 0
        for i in range(n):
            result += (xi[i] - self._q[i]) ** 2 - (xi[i] - self._p[i]) ** 2

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _find_split(self, float [:, :] X):

        # Calculate similarities
        cdef int n = X.shape[0]
        cdef np.ndarray[np.float32_t, ndim=1] array = np.zeros(n, dtype=np.float32, order='c')
        cdef float [:] similarities = array


        cdef int num_threads = 4
        if n < 12:
            num_threads = 1
        cdef int i = 0
        # Read about different schedules https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html
        for i in prange(n, schedule='dynamic', nogil=True, num_threads=num_threads):
            array[i] = self.sqeuclidean_projection(X[i])

        cdef float similarities_min = np.min(array)
        cdef float similarities_max = np.max(array)

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

    cdef int distance(self, float [:] xi, float [:] xj) nogil:
        if self.is_leaf:
            return self.depth

        cdef bint path_i = self.sqeuclidean_projection(xi) <= self._split_point
        cdef bint path_j = self.sqeuclidean_projection(xj) <= self._split_point

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
        cdef np.ndarray[np.float32_t, ndim=1] distance_matrix = np.ones(<int>comb(n, 2), dtype=np.float32, order='c')
        cdef float [:] view = distance_matrix


        cdef int diagonal = 1
        cdef int idx = 0
        for c in range(n):
            for r in range(diagonal, n):
                view[idx] = 1 / <float>self.distance(X[c], X[r])
                idx += 1
            diagonal += 1

        return distance_matrix

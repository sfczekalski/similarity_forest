cimport cython

cdef float dot_projection(float [:] xi, float [:] p, float [:] q) nogil
cdef float rbf_projection(float [:] xi, float [:] p, float [:] q) nogil
cdef float sqeuclidean_projection(float [:] xi, float [:] p, float [:] q) nogil
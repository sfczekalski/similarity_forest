cimport cython
import numpy as np
cimport numpy as np

cdef float [:] dot_projection(float [:, :] X, float [:] p, float [:] q, float [:] result) nogil
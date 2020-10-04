cimport cython
import numpy as np
cimport numpy as np

cdef np.ndarray[dtype=np.float32_t, ndim=1] dot_projection(float [:, :] X, float [:] p, float [:] q)
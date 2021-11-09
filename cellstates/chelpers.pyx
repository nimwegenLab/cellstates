# cython: language_level=3, boundscheck=False

cimport cython
import numpy as np
cimport numpy as np
from cpython.exc cimport PyErr_CheckSignals
from .cluster cimport Cluster, find_cluster_distance


cpdef np.ndarray[np.float_t, ndim = 1] get_cluster_distances(Cluster clst):
    """
    returns condensed matrix X of log-likelihood changes when pairs
    of clusters are merged. -X gives the distance in the traditional sense
    (large +ve values means dissimilarity), but there may be distances < 0.
    """
    cdef:
        int i, j
        double distance
        int Nb = clst.N_boxes
        int idx = 0
        np.ndarray[np.float_t, ndim = 1] X = np.zeros(Nb*(Nb-1)/2,
                                                      dtype=np.float64)
    for i in range(Nb):
        for j in range(i+1, Nb):
            PyErr_CheckSignals()
            if clst._cluster_sizes[i] and clst._cluster_sizes[j]:
                distance = find_cluster_distance(clst, i, j)
                X[idx] = distance
            idx += 1

    return X

cimport numpy as np
from .cluster cimport Cluster


cpdef np.ndarray[np.float_t, ndim = 1] get_cluster_distances(Cluster clst)

# cython: language_level=3, boundscheck=False, wraparound=False

cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from cpython.exc cimport PyErr_CheckSignals
from .cluster cimport Cluster, find_cluster_distance
from scipy.special.cython_special cimport betainc, logit


cpdef np.ndarray[np.float_t, ndim = 1] get_cluster_distances(Cluster clst):
    """
    returns condensed matrix X of log-likelihood changes when pairs
    of clusters are merged. -X gives the distance in the traditional sense
    (large +ve values means dissimilarity), but there may be distances < 0.

    Parameters
    ----------
    clst : cellstate.Cluster object

    Returns
    -------
    X : cluster distance table
    """
    cdef:
        int i, j
        double distance
        int Nb = clst.N_boxes
        int idx = 0
        np.ndarray[np.float_t, ndim = 1] X = np.zeros(Nb*(Nb-1)//2,
                                                      dtype=np.float64)
    for i in range(Nb):
        for j in range(i+1, Nb):
            PyErr_CheckSignals()
            if clst._cluster_sizes[i] and clst._cluster_sizes[j]:
                distance = find_cluster_distance(clst, i, j)
                X[idx] = distance
            idx += 1

    return X

@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float_t, ndim = 1] marker_scores(Cluster clst, list C1, list C2):
    """
    Marker gene scores that predict how well a gene can separate the two groups of cellstates
    C1 and C2. The score is positive if the gene expression is higher in C2 than C1 and negative
    otherwise.

    Parameters
    ----------
    clst : cellstate.Cluster object
    C1 : list of cellstate indices
    C2 : list of cellstate indices

    Returns
    -------
    gene_scores : numpy array of floats of length G (the number of genes)
    """
    cdef:
        int i, j, c1, c2, g
        double x, a, b, P
        double C1C2, weight
        double[:] gene_scores
    gene_scores = np.zeros(clst.G, dtype=float)
    C1C2 = np.sum(clst.cluster_sizes[C1])*np.sum(clst.cluster_sizes[C2])

    for c1 in C1:
        for c2 in C2:
            PyErr_CheckSignals()
            weight = clst._cluster_sizes[c1]*clst._cluster_sizes[c2] / C1C2
            for g in prange(clst.G, nogil=True, num_threads=clst.num_threads):
                a = clst._cluster_umi_counts[g, c1] + clst.LAMBDA[g]
                b = clst._cluster_umi_counts[g, c2] + clst.LAMBDA[g]
                x = (clst._cluster_umi_sum[c1] + clst.LAMBDA_sum) / \
                        (clst._cluster_umi_sum[c1] + clst._cluster_umi_sum[c2] + 2*clst.LAMBDA_sum)
                P = betainc(a, b, x)*weight
                gene_scores[g] += P

    for g in range(clst.G):
        gene_scores[g] = logit(gene_scores[g])

    return np.asarray(gene_scores, dtype=float)

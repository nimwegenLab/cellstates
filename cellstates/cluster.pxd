cdef class Cluster:
    cdef readonly:
        int G  # number of genes
        int N_samples  # number of cells
        int N_boxes  # number of possible clusters
    cdef:
        long[:, :] data  # sc expression data
        # total UMI counts per gene for each cluster
        long[:, :] _cluster_umi_counts
        double[:] LAMBDA  # dirichlet prior
        double LAMBDA_sum  # sum of dirichlet prior
        double B  # dirichlet prior normalisation constant
        int[:] _clusters  # clusters[i] = j means cell i is in cluster j
        double[:] _likelihood  # log-likelihood of every cluster
        int[:] _cluster_sizes  # number of cells in every cluster
        long[:] _cluster_umi_sum  # total UMI counts per cluster
        long[:] _cell_umi_sum  # total UMI counts per cell
        double[:, :] _lngamma_cache
        int n_cache
    cdef void _init_lngamma_cache(self)
    cdef void _init_counts(self)
    cdef void _init_likelihood(self)
    cpdef void combine_two_clusters(self, int c1, int c2)
    cpdef get_best_move(self, int m, move_to=*)
    cpdef int optimal_move(self, int m, move_to=*)
    cpdef void move_cell(self, int m, int c_new)

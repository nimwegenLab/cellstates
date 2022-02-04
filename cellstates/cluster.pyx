#cython: language_level=3, boundscheck=False

cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from cpython.exc cimport PyErr_CheckSignals
from libc.stdlib cimport rand, srand


# ------ c-functions and constants ------ #

@cython.cdivision(True)
cdef double randnum():
    """ returns random float between 0 and 1 """
    return rand() / <double>RAND_MAX

@cython.cdivision(True)
cdef int randint(int n):
    """ returns random int between 0 and n-1 """
    return rand() % n


# ------ functions for setting log-gamma cache ------ #

cdef double[:, :] LNGAMMA_CACHE
cdef int N_CACHE


def set_cache(double[:, :] cache):
    global LNGAMMA_CACHE
    global N_CACHE
    LNGAMMA_CACHE = cache
    N_CACHE = cache.shape[1]



cdef double lgamma_cache(double[:] LAMBDA, int g, long n) nogil:
    if n < N_CACHE:
        return LNGAMMA_CACHE[g, n]
    else:
        return lgamma(LAMBDA[g] + n)


# ------ likelihood functions ------ #


cdef double find_dirichlet_norm(double[:] LAMBDA, int num_threads):
    cdef:
        int G = LAMBDA.shape[0]
        double thesum = 0
        double B = 0
        int i

    for i in prange(G, nogil=True, num_threads=num_threads):
        thesum += LAMBDA[i]
        B -= lgamma(LAMBDA[i])
    B += lgamma(thesum)

    return B



cdef double find_LL_c_old(Cluster clst, int m):
    """ find new log-likelihood when cell m is moved out of its cluster"""
    cdef:
        double result
        long n
        long[:] C, M
        double[:] lmbd
        int g
        int c_old = clst._clusters[m]

    if clst._cluster_sizes[c_old] > 1:
        C = clst._cluster_umi_counts[:, c_old]
        M = clst.data[:, m]
        lmbd = clst.LAMBDA

        n = clst._cluster_umi_sum[c_old] - clst._cell_umi_sum[m]
        result = clst.B - lgamma(n + clst.LAMBDA_sum)
        for g in prange(clst.G, nogil=True, num_threads=clst.num_threads):
            n = C[g] - M[g]
            result += lgamma_cache(lmbd, g, n)
    else:
        result = 0.
    return result



cdef double find_LL_c_new(Cluster clst, int m, int c_new):
    """ find new log-likelihood when cell m is moved out into cluster c_new"""
    cdef:
        double result
        long n
        long[:] C, M
        double[:] lmbd
        int g

    C = clst._cluster_umi_counts[:, c_new]
    M = clst.data[:, m]
    lmbd = clst.LAMBDA

    n = clst._cluster_umi_sum[c_new] + clst._cell_umi_sum[m]
    result = clst.B - lgamma(n + clst.LAMBDA_sum)

    for g in prange(clst.G, nogil=True, num_threads=clst.num_threads):
        n = C[g] + M[g]
        result += lgamma_cache(lmbd, g, n)
    return result



cdef double find_cluster_LL(Cluster clst, int c):
    """ find log-likelihood of cluster c """
    cdef:
        double result
        int g
        long[:] C = clst._cluster_umi_counts[:, c]
        double[:] lmbd = clst.LAMBDA

    if clst._cluster_sizes[c] > 0:
        result = clst.B - lgamma(clst._cluster_umi_sum[c] + clst.LAMBDA_sum)
        for g in prange(clst.G, schedule='static', nogil=True, num_threads=clst.num_threads):
            result += lgamma_cache(lmbd, g, C[g])
    else:
        result = 0.

    return result



cdef double find_cluster_distance(Cluster clst, int i, int j):
    """ return change in likelihood if two clusters i and j were merged """
    cdef:
        double delta
        long n
        long[:] Ci, Cj
        double[:] lmbd
        int g
    Ci = clst._cluster_umi_counts[:, i]
    Cj = clst._cluster_umi_counts[:, j]
    lmbd = clst.LAMBDA

    n = clst._cluster_umi_sum[i] + clst._cluster_umi_sum[j]
    delta = clst.B - lgamma(n + clst.LAMBDA_sum)

    for g in prange(clst.G, nogil=True, num_threads=clst.num_threads):
        n = Ci[g] + Cj[g]
        delta += lgamma_cache(lmbd, g, n)
    delta -= clst._likelihood[i] + clst._likelihood[j]
    return delta


# ------ moving cells between clusters ------ #


cdef void move_cell_nocalc(Cluster clst, int m, int c_new,
                            double LL_c_old, double LL_c_new):
    """ move cell m to cluster c_new in Cluster object;
    likelihood changes need to be pre-calculated """
    cdef int c_old = clst._clusters[m]

    if c_old != c_new:
        clst._cluster_sizes[c_old] -= 1
        clst._clusters[m] = c_new
        clst._cluster_sizes[c_new] += 1
        clst._likelihood[c_old] = LL_c_old
        clst._likelihood[c_new] = LL_c_new

        clst._cluster_umi_sum[c_old] -= clst._cell_umi_sum[m]
        clst._cluster_umi_sum[c_new] += clst._cell_umi_sum[m]

        for g in range(clst.G):
            clst._cluster_umi_counts[g, c_old] -= clst.data[g, m]
            clst._cluster_umi_counts[g, c_new] += clst.data[g, m]


cdef void move_cell(Cluster clst, int m, int c_new):
    cdef double LL_c_old, LL_c_new
    cdef int c_old = clst._clusters[m]
    if c_old != c_new:
        LL_c_old = find_LL_c_old(clst, m)
        LL_c_new = find_LL_c_new(clst, m, c_new)
        move_cell_nocalc(clst, m, c_new, LL_c_old, LL_c_new)


cdef void merge_two_clusters(Cluster clst, int c1, int c2, delta=None):
    """ merge the two clusters c1 and c2 into one (label c1) """
    cdef:
        double delta_LL
        int i

    if delta is None:
        delta_LL = find_cluster_distance(clst, c1, c2)
    else:
        delta_LL = <double > delta

    for i in range(clst.N_samples):
        if clst._clusters[i] == c2:
            clst._clusters[i] = c1

    clst._cluster_sizes[c1] += clst._cluster_sizes[c2]
    clst._cluster_sizes[c2] = 0

    clst._likelihood[c1] += delta_LL + clst._likelihood[c2]
    clst._likelihood[c2] = 0.

    for i in range(clst.G):
        clst._cluster_umi_counts[i, c1] += clst._cluster_umi_counts[i, c2]
        clst._cluster_umi_counts[i, c2] = 0

    clst._cluster_umi_sum[c1] += clst._cluster_umi_sum[c2]
    clst._cluster_umi_sum[c2] = 0


# ------ running MCMC ------ #


cdef int do_biased_mc_moves(Cluster clst, int N_steps,
                            int tries_per_step=500, int min_index=0) except -1:
    """ uniform sampling of partition space with bias towards better partitions
    returns total number of attempted moves
    """
    cdef:
        int max_tries = N_steps*tries_per_step
        int n_move_successes = 0, tries = 0
        int n_rand_max = clst.N_samples - min_index
        int m, c_new, c_old
        int n_empty
        double move_bias, move_likelihood
        double LL_c_new, LL_c_old

    set_cache(clst._lgamma_cache)

    while tries < max_tries and n_move_successes < N_steps:
        PyErr_CheckSignals()

        m = randint(n_rand_max) + min_index  # index of cell
        c_new = randint(clst.N_boxes)  # index of new cluster
        c_old = clst._clusters[m]  # index of current cluster of m

        # Biase move to generate uniform samplng of partition space
        if c_old == c_new:
            # no move
            continue
        elif clst._cluster_sizes[c_new] == 0:
            if clst._cluster_sizes[c_old] == 1:
                # no change in partition
                continue
            else:
                # n_clusters += 1
                move_bias = - log(clst.N_boxes - clst.n_clusters)
        elif clst._cluster_sizes[c_old] == 1 and clst._cluster_sizes[c_new] != 0:
            # n_clusters -= 1
            move_bias = log(clst.N_boxes - clst.n_clusters + 1.)
        else:
            # n_clusters unchanged
            move_bias = 0.

        # only count "proper" moves
        tries += 1

        # find change in LL and accept move with probability
        # P_bias*P_new/P_old if it isn't
        LL_c_new = find_LL_c_new(clst, m, c_new)
        LL_c_old = find_LL_c_old(clst, m)
        move_likelihood = move_bias + (LL_c_old + LL_c_new) - \
            (clst._likelihood[c_old] + clst._likelihood[c_new])

        if move_likelihood < MIN_EXP_ARG:
            # if delta_LL is too small, exp will fail
            # probability of move is too small to be represented by
            # double and hence rejected
            continue
        elif move_likelihood < 0 and randnum() > exp(move_likelihood):
            # accept move only with probability exp(move_likelihood)
            continue

        # move gets accepted
        move_cell_nocalc(clst, m, c_new, LL_c_old, LL_c_new)
        n_move_successes += 1

    if n_move_successes < N_steps:
        raise RuntimeError((f'Only {n_move_successes} moves found within loop limit.'
                            'Consider raising tries_per_step'))
    return tries


# ------ merging/cluster hierarchy ------ #


cdef tuple find_argsmax(double[:, :] D, unsigned char[:] mask):
    """
    find index of maximal value in D assuming D is diagonal and only
    considering the rows specified by the mask

    Parameters
    ----------
    D : double[:, :]
        diagonal matrix in which to find the position of the maximum
    mask : unsigned char[:]
        boolean array to indicate which positions to consider

    Returns
    -------
    D_max : double
        maximum value found in matrix
    j_max : int
        smaller index of maximum position
    i_max : int
        larger index of maximum position
    """
    cdef:
        int N = D.shape[0]
        double D_max = -np.inf
        int i_max, j_max, i, j

    for i in range(N):
        if mask[i]:
            for j in range(i):
                if D[i, j] > D_max and mask[j]:
                    D_max = D[i, j]
                    i_max = i
                    j_max = j
    # note that j < i
    return D_max, j_max, i_max



def merge_clusters_hierarchical(Cluster clst,
                                double LL_threshold=0.,
                                int n_cluster_threshold=1):
    """ perform hierarchical merging of clusters """
    cdef:
        int i, j, k, _
        double delta
        cdef list delta_LL_history, merge_hierarchy

    set_cache(clst._lgamma_cache)

    # boolean array indicating which clusters still exist
    cdef np.ndarray[np.uint8_t, ndim = 1] cluster_exists
    cluster_exists = np.asarray(clst.cluster_sizes > 0, dtype=np.uint8)

    # 2D array  that  contains  change  in  log-likelihood
    # for  the  merge of any of two clusters
    cdef np.ndarray[np.float_t, ndim = 2] delta_LL
    delta_LL = np.zeros((clst.N_boxes, clst.N_boxes), dtype=np.float64)

    for i in range(clst.N_boxes):
        PyErr_CheckSignals()

        if not cluster_exists[i]:
            continue
        for j in range(i):
            if not cluster_exists[j]:
                continue

            delta = find_cluster_distance(clst, i, j)

            delta_LL[i, j] = delta
            delta_LL[j, i] = delta

    # array of cluster label
    cdef np.ndarray[np.int_t, ndim = 1] clusters = np.arange(clst.N_boxes)

    delta_LL_history = []
    merge_hierarchy = []

    for _ in range(clst.n_clusters - n_cluster_threshold):
        PyErr_CheckSignals()

        # start by making new clusters based on lowest value in delta_LL and
        delta, i, j = find_argsmax(delta_LL, cluster_exists)

        # stop  clustering  if  there  are  no  favourable  changes  left
        if delta <= LL_threshold:
            break

        merge_hierarchy.append((i, j))
        delta_LL_history.append(delta)

        merge_two_clusters(clst, i, j, delta=delta)
        cluster_exists[j] = False

        # update values of best_delta_LL and best_cluster
        for k in range(clst.N_boxes):
            delta_LL[j, k] = 0.
            delta_LL[k, j] = 0.

            if cluster_exists[k] and k != i:
                delta = find_cluster_distance(clst, i, k)
                delta_LL[i, k] = delta
                delta_LL[k, i] = delta

    return merge_hierarchy, delta_LL_history



def merge_clusters_optimally(Cluster clst):
    """ merge clusters hierarchically until maximum likelihood """
    cdef:
        int a_max, c1, c2
        list cluster_hierarchy, delta_LL_history
        np.ndarray[np.float_t, ndim = 1] total_delta
    cluster_hierarchy, delta_LL_history = clst.get_cluster_hierarchy()

    # find maximum in total LL change after each merge
    total_delta = np.cumsum(delta_LL_history)
    a_max = np.argmax(total_delta)

    # do optimal merges if they increase total LL
    if total_delta[a_max] > 0.:
        for c1, c2 in cluster_hierarchy[:a_max + 1]:
            merge_two_clusters(clst, c1, c2)


# ------ optimizing cluster ------ #


def optimize_cell_positions_full(Cluster clst):
    """
    move individual cells to their optimal cluster.
    performs multiple rounds of optimization until all
    cells are optimally placed.
    """

    cdef:
        int cell, c_best, move_count = 0
        np.ndarray[np.int_t, ndim = 1] cell_iter
        np.ndarray[np.float_t, ndim = 1] best_delta_LL
        double delta

    cell_iter = np.random.permutation(clst.N_samples)
    best_delta_LL = np.zeros(clst.N_samples, dtype=np.float64)
    while True:
        PyErr_CheckSignals()
        for cell in cell_iter:
            c_best, delta = clst.get_best_move(cell)
            if delta > 0.:
                move_cell(clst, cell, c_best)
                move_count += 1

                # find new best_delta_LL
                c_best, delta = clst.get_best_move(cell)

            best_delta_LL[cell] = delta

        if move_count == 0:
            # None of the cells changed cluster
            break
        else:
            # initialize next round of moves
            move_count = 0
            # only 20% of cells are considered in next round, the assumption
            # being that most cells are already well placed
            cell_iter = np.argsort(-best_delta_LL)[:clst.N_samples//5]



def optimize_cell_positions_simple(Cluster clst):
    """
    move individual cells to their optimal cluster
    performs only a single round of optimization
    """

    cell_iter = np.random.permutation(clst.N_samples)
    for cell in cell_iter:
        c_best = clst.optimal_move(cell)


# ------ Cluster class definition ------ #


cdef class Cluster:
    """
    Object that contains SC expression data, lambda and stores a partition

    Parameters
    ----------
    d : 2D array of ints
        UMI count array of shape (N_genes, N_cells)
    l : 1D array or float, default=None
        If array, the pseudo-counts lambda for each gene. Has to be of shape (N_genes, ).
        All entries must be > 0. If float, defines the magnitude of the pseudo-counts,
        but their relative sizes is set by data average. If None, magnitude is chosen
        automatically.
    c : 1D array of ints, default=None
        Cluster labels for each cell. Has to be of shape (N_cells, ).
        Labels must be positive and < N_boxes. If None, every cell is in its own cluster.
    genes : 1D array, optional
        Names of genes; does not have to be set.
    max_clusters : int, default 0
        Maximum number of clusters allowed (acces through item N_boxes).
        if max_clusters=0, N_boxes=N_cells
    num_threads : int, default 1
        Number of threads used in parallel computations.
    n_cache : int, default=100
        Number of lgamma values to store per gene.
        Large values speed up calculations, but use more memory
    seed : unsigned int, default=1
        random generator seed

    Returns
    -------
    clst : object of type Cluster
    """

    def __cinit__(self, d,
                  l=None,
                  c=None,
                  genes=None,
                  int max_clusters=0,
                  int num_threads=1,
                  int n_cache=100,
                  unsigned int seed=1):

        if isinstance(l, np.ndarray):
            if d.shape[0] != l.shape[0]:
                raise ValueError('The shapes of the data and lambda do not match')
            elif np.any(l<=0):
                raise ValueError('all dirichlet pseudo-counts must be >0')
            self.LAMBDA_sum = np.sum(l)
        else:
            if l is None:
                self.LAMBDA_sum = 2**(np.round(np.log2(d.sum()/d.shape[1])))
            else:
                l = float(l)
                if l <= 0.:
                    raise ValueError('dirichlet prior parameter must be > 0')
                self.LAMBDA_sum = l
            l = self.LAMBDA_sum*np.sum(d, axis=1)/np.sum(d)
            mask = ( l>0 )
            if np.any(mask):
                l = l[mask]
                d = d[mask, :]

        self.LAMBDA = <np.ndarray[np.float_t, ndim = 1]> l
        self.data = <np.ndarray[np.int_t, ndim = 2]?> d
        self.G = d.shape[0]
        self.N_samples = d.shape[1]

        if max_clusters > 0:
            self.N_boxes = max_clusters
        else:
            self.N_boxes = self.N_samples

        if c is None:
            c = np.arange(self.N_samples, dtype=np.int32)
        elif d.shape[1] != len(c):
            raise ValueError(
                'the shapes of the data and clusters do not match')
        else:
            # check clusters provided is consistent with N_boxes
            if np.max(c) >= self.N_boxes:
                raise ValueError(
                    'all cluster labels must be smaller than max_clusters')
            elif np.min(c) < 0:
                raise ValueError('all cluster labels must be positive')
            c = np.array(c, dtype=np.int32)

        self._clusters = <np.ndarray[np.int32_t, ndim = 1]?> c

        self.n_cache = n_cache
        self.num_threads = num_threads

        srand(seed)

    def __init__(self, d,
                  l=None,
                  c=None,
                  genes=None,
                  int max_clusters=0,
                  int num_threads=1,
                  int n_cache=100,
                  unsigned int seed=1):

        if genes is not None:
            if not isinstance(l, np.ndarray):
                mask = np.any(d, axis=1)
                self.genes = genes[mask]
            else:
                self.genes = genes

        self._init_lgamma_cache()

        self._init_counts()
        self.B = find_dirichlet_norm(self.LAMBDA, self.num_threads)
        self._init_likelihood()

    cdef void _init_lgamma_cache(self):
        cdef int i, g
        self._lgamma_cache = np.zeros((self.G, self.n_cache), dtype=np.float64)
        for g in range(self.G):
            for i in range(self.n_cache):
                self._lgamma_cache[g, i] = lgamma(self.LAMBDA[g] + i)
        set_cache(self._lgamma_cache)

    cdef void _init_counts(self):
        """
        initialize several matrices:
            - _cluster_sizes: size of each cluster
            - _cluster_umi_counts: total UMI counts for each gene per cluster
            - _cluster_umi_sum: total UMI counts per cluster
            - _cell_umi_sum: total UMI counts per cell
        for aggregate UMI counts in each cluster and cell
        """
        cdef:
            int i, c, g
            long n, N

        # size of each cluster
        self._cluster_sizes = np.zeros(self.N_boxes, dtype=np.int32)
        # total counts per gene in each cluster
        self._cluster_umi_counts = np.zeros((self.G, self.N_boxes),
                                            dtype=np.int64)

        # total counts in each cluster
        self._cluster_umi_sum = np.zeros(self.N_boxes, dtype=np.int64)
        # total counts per cell
        self._cell_umi_sum = np.zeros(self.N_samples, dtype=np.int64)

        for i in range(self.N_samples):
            c = self._clusters[i]
            self._cluster_sizes[c] += 1
            N = 0
            for g in range(self.G):
                n = self.data[g, i]
                self._cluster_umi_counts[g, c] += n
                N += n
            self._cell_umi_sum[i] = N
            self._cluster_umi_sum[c] += N

    cdef void _init_likelihood(self):
        """ calculate all cluster likelihoods """
        cdef:
            int c
        self._likelihood = np.zeros(self.N_boxes, dtype=np.float64)
        for c in range(self.N_boxes):
            self._likelihood[c] = find_cluster_LL(self, c)

    def biased_monte_carlo_sampling(self, int N_steps=1, int tries_per_step=1000,
                                    int min_index=0, int N_batch=0):
        """
        Do N_steps steps of Monte-Carlo sampling of partition space where the
        move-set is biased as towards higher likelihood partitions.

        Parameters
        ----------
        N_steps : int, default=1
            Number of MCMC steps to run.
        tries_per_step : int, default=1000
            Number of moves proposed per step before RuntimeError is raised.
        min_index : int, default=0
            Only move cells with index min_index to N_samples-1
        N_batch : int, default=0
            Deprecated; Included for backward compatibility; use tries_per_step
            A maximum of 100*N_batch*N_steps moves can be proposed in total
            before a RuntimeError is raised.

        Returns
        -------
        The Cluster object is updated to contain a new partition

        total_tries : int
            Number of proposed moves until N_steps were accepted.

        Raises
        ------
        RuntimeError:
            If too many moves are rejected, a RuntimeError is raised.
        """
        if N_batch > 0:
            tries_per_step = 100*N_batch*N_steps
        return do_biased_mc_moves(self, N_steps, tries_per_step, min_index)

    def set_N_boxes(self, int Nb_new):
        """
        change possible number of clusters.
        clusters in Cluster object are automatically relabelled from 0 to
        (n_clusters-1) and mapping is returned

        Parameters
        ----------
        Nb_new : int
            New maximum number of clusters allowed.

        Returns
        -------
        mapping : dict
            Dictionary mapping old cluster labels to new ones.
            mapping[i_new] = i_old where i_new is the new label for cluster
            i_old
        """

        if Nb_new < self.n_clusters:
            raise ValueError(
                "Nb_new must be larger or equal to the number of clusters")

        cdef int i, c
        cdef dict mapping = {i: c for i, c in
                             enumerate(np.sort(np.unique(self.clusters)))}
        self.N_boxes = Nb_new

        cdef:
            np.ndarray[np.int32_t, ndim = 1] \
                new_clusters = np.zeros(self.N_samples, dtype=np.int32)
            np.ndarray[np.float_t, ndim = 1] \
                new_likelihood = np.zeros(Nb_new, dtype=np.float64)

        for i in range(self.n_clusters):
            c = mapping[i]
            new_likelihood[i] = self._likelihood[c]
            new_clusters[self.clusters == c] = i

        self._clusters = new_clusters
        self._likelihood = new_likelihood
        self._init_counts()

        return mapping

    def set_dirichlet_pseudocounts(self, l, int n_cache=-1):
        """
        Set dirichlet pseudocounts (lambda) to new values. Optionally also set n_cache new.

        Parameters
        ----------
        l : 1D array or float
            If array, the new pseudo-counts lambda for each gene. Has to be of shape (N_genes, ).
            All entries must be > 0. If float, defines the magnitude of the pseudo-counts,
            but their relative sizes is set by data average.
        n_cache : int, default=-1
            Number of lgamma values to store per gene.
            If < 0, n_cache is not changed.
            Large values speed up calculations, but use more memory

        Raises
        ------
        ValueError
            when shape is mismatched to data, or values are <=0.
        """
        if isinstance(l, np.ndarray):
            if self.G != l.shape[0]:
                raise ValueError('The shapes of the data and lambda do not match')
            elif np.any(l<=0):
                raise ValueError('all dirichlet pseudo-counts must be >0')
            self.LAMBDA_sum = np.sum(l)
        else:
            l = float(l)
            if l <= 0.:
                raise ValueError('dirichlet prior parameter must be > 0')
            self.LAMBDA_sum = l
            l = self.LAMBDA_sum*np.sum(self.umi_data, axis=1)/np.sum(self.umi_data)

        self.LAMBDA = <np.ndarray[np.float_t, ndim = 1]> l
        self.B = find_dirichlet_norm(self.LAMBDA, self.num_threads)
        if n_cache > 0:
            self.n_cache = n_cache
        self._init_lgamma_cache()
        self._init_likelihood()

    def set_clusters(self, new_clusters, int max_clusters=0):
        """
        Set clusters to new values.

        Parameters
        ----------
        new_clusters : array-like
            new clusters
        max_clusters : int, default 0
            Maximum number of clusters allowed (acces through item N_boxes).
            if max_clusters=0, N_boxes=max(new_clusters)

        Raises
        ------
        ValueError
            when entries in new_clusters are negative
        """

        # check clusters provided is consistent with N_boxes
        if np.min(new_clusters) < 0:
            raise ValueError('all cluster labels must be positive')

        cdef int max_label
        max_label = np.max(new_clusters)

        if max_clusters > max_label:
            self.set_N_boxes(max_clusters)
        else:
            if max_label >= self.N_boxes:
                self.set_N_boxes(max_label+1)

        c = np.array(new_clusters, dtype=np.int32)

        self._clusters = <np.ndarray[np.int32_t, ndim = 1]?> c
        self._init_counts()
        self._init_likelihood()

    def merge_clusters(self, double LL_threshold=0.,
                         int n_cluster_threshold=1):
        """
        Method to do hierarchical merging of clusters. Default settings stop
        when optimum log-likelihood is reached.

        Parameters
        ----------
        LL_threshold : float, default=0.
            Threshold by how much log-likelihood must change to merge clusters.
            If LL_threshold < 0, we also consider moves that reduce total LL.
            If LL_threshold = -np.if, clusters will be merged until
            n_clusters_threshold is reached.
        n_clusters_threshold : int, default=1
            Minimum number of clusters allowed after merging. Must be > 0

        Returns
        -------
        hierarchy : list of tuples
            List of order in which clusters were merged
        delta_LL_history : list of floats
            List of change in log-likelihood associated with each merging step.

        Raises
        ------
        ValueError
            When n_cluster_threshold < 1
        """

        if n_cluster_threshold < 1:
            raise ValueError('n_cluster_threshold must be > 0')

        return merge_clusters_hierarchical(self, LL_threshold,
                                           n_cluster_threshold)

    def get_cluster_hierarchy(self):
        """
        Function to get complete hierarchical tree of cluster similarities.
        Unlike merge_clusters, it does not change the state of the Cluster
        object.

        Returns
        -------
        hierarchy : list of tuples
            List of order in which clusters were merged
        delta_LL_history : list of floats
            List of change in log-likelihood associated with each merging step.
        """
        c = self.clusters.copy()
        merge_hierarchy, delta_LL_history = \
            merge_clusters_hierarchical(self, LL_threshold=-np.inf,
                                        n_cluster_threshold=1)
        self.set_clusters(c)

        return merge_hierarchy, delta_LL_history

    cpdef void combine_two_clusters(self, int c1, int c2):
        """
        combines two clusters labelled c1 and c2 into one cluster labelled c1.

        Parameters
        ----------
        c1, c2 : int
            labels of clusters to combine into one.
            The combined cluster has label c1.

        Returns
        -------
        None
            Updates Cluster object
        """
        merge_two_clusters(self, c1, c2)

    cpdef get_best_move(self, int m, move_to=None):
        """
        Find best move for cell m. Returns the best cluster to move m to and
        the associated change in log-likelihood, but does not perform move.
        In case m is already in the optimal cluster, the second best option
        is given (with negative best_delta_LL).
        If an iterable move_to is given, only moves to those clusters will
        be considered.

        Parameters
        ----------
        m : int
            cell index
        move_to : iterable of ints, optional
            iterable of all clusters to consider moving cell m to.
            Default is to consider all available clusters.

        Returns
        -------
        c_best : int
            cluster to which cell was moved
        best_delta_LL : flaot
            change in total log-likelihood associated with move
        """

        cdef:
            int c_new
            int c_old = self._clusters[m]
            int c_best = c_old
            double delta_LL, best_delta = -np.inf
            double LL_c_old = find_LL_c_old(self, m)
            double LL_c_new

        if move_to is None:
            move_to = range(self.N_boxes)

        for c_new in move_to:
            if c_new == c_old:
                continue
            LL_c_new = find_LL_c_new(self, m, c_new)
            delta_LL = (LL_c_old + LL_c_new) \
                - (self._likelihood[c_new] + self._likelihood[c_old])

            if delta_LL > best_delta:
                best_delta = delta_LL
                c_best = c_new

        return c_best, best_delta

    cpdef int optimal_move(self, int m, move_to=None):
        """
        Move cell m to its best cluster. Returns the new cluster of m.
        If an iterable move_to is given, only moves to those clusters will
        be considered.

        Parameters
        ----------
        m : int
            cell index
        move_to : iterable, optional
            iterable of all clusters to consider moving cell m to.
            Default is to consider all available clusters.

        Returns
        -------
        c_best : int
            cluster to which cell was moved
        """

        cdef:
            int c_best
            double delta_LL

        c_best, delta_LL = self.get_best_move(m, move_to)

        if delta_LL > 0.:
            self.move_cell(m, c_best)
        else:
            c_best = self._clusters[m]

        return c_best

    def optimize_clusters(self, bint merge_clusters=True,
                          bint optimize_cells=True,
                          bint optimize_simple=True,
                          bint set_N_boxes=True):
        """
        Method to optimize current partition without further MCMC steps.
        Clusters are first optimally merged then individual cells are moved
        to their optimal cluster.
        Cluster labels and N_boxes may change in the process.

        Parameters
        ----------
        merge_clusters : bool, default=True
            whether to include the cluster-merging step
        optimize_cells : bool, default=True
            whether to include the cell-moving step
        optimize_simple : bool, default=True
            If True, every cell's position is optimized once.
            If False, several rounds of optimization are performed.
        set_N_boxes : bool, default = True
            whether to allow N_boxes to be changed
        """
        if merge_clusters:
            merge_clusters_optimally(self)

        if set_N_boxes:
            self.set_N_boxes(self.n_clusters + 2)

        if optimize_cells:
            if optimize_simple:
                optimize_cell_positions_simple(self)
            else:
                optimize_cell_positions_full(self)

    cpdef void move_cell(self, int m, int c_new):
        """
        move cell m into cluster c_new

        Parameters
        ----------
        m : int
            index of cell to move
        c_new : int
            index of cluster to move cell into
        """
        move_cell(self, m, c_new)

    def get_expressionstate(self, int c):
        """
        Get modal expression state vector of cluster c.

        Parameters
        ----------
        c : int
            cluster index

        Returns
        -------
        f : array of floats
            modal gene expression state vector

        Raises
        ------
        ValueError
            When c is an empty cluster
        """
        cdef:
            np.ndarray[np.float_t, ndim = 1] f
        if self._cluster_sizes[c] == 0:
            raise ValueError(f'{c} is an empty cluster')
        else:
            f = self.cluster_umi_counts[:, c] \
                + np.asarray(self.LAMBDA) - 1.
            f[f < 0.] = 0.
            f = f/np.sum(f)
        return f

    def get_expressionstate_mv(self, int c):
        """
        Get mean and variance of expression state vector
        They are calculated independently for each gene and hence does not
        add up to 1.

        Parameters
        ----------
        c : int
            cluster index

        Returns
        -------
        f : array of floats
            mean gene expression state vector
        var_f : array of floats
            predicted variance of each gene

        Raises
        ------
            ValueError
                When c is an empty cluster
        """

        cdef:
            np.ndarray[np.float_t, ndim = 1] f
            np.ndarray[np.float_t, ndim = 1] var_f
            np.ndarray[np.float_t, ndim = 1] n_gc
            double n_c
        if self._cluster_sizes[c] == 0:
            raise ValueError(f'{c} is an empty cluster')
        else:
            n_gc = self.cluster_umi_counts[:, c] \
                + np.asarray(self.LAMBDA)
            n_c = np.sum(n_gc)
            f = n_gc/n_c
            var_f = (f*(1 - f))/(n_c + 1.)
        return f, var_f

    @property
    def n_clusters(self):
        """ number of clusters """
        return np.count_nonzero(self.cluster_sizes)

    @property
    def total_likelihood(self):
        """ total log-likelihood of partition """
        cdef double thesum = 0.
        for i in range(self.N_boxes):
            thesum += self._likelihood[i]
        return thesum

    @property
    def likelihood(self):
        """ array of log-likelihoods for every cluster """
        return np.asarray(self._likelihood, dtype=np.float64)

    @property
    def clusters(self):
        """ array specifying clusters.
        clusters[i] = j means cell i is in cluster j """
        return np.asarray(self._clusters, dtype=np.int32)

    @property
    def cluster_sizes(self):
        """ array of cluster sizes """
        return np.asarray(self._cluster_sizes, dtype=int)

    @property
    def cluster_umi_counts(self):
        """ array of total UMI counts per gene in each cluster """
        return np.asarray(self._cluster_umi_counts, dtype=int)

    @property
    def cluster_umi_sum(self):
        """ array of total UMI counts in each cluster """
        return np.asarray(self._cluster_umi_sum, dtype=int)

    @property
    def dirichlet_pseudocounts(self):
        """ array of dirichlet prior pseudocounts """
        return np.asarray(self.LAMBDA, dtype=float)

    @property
    def umi_data(self):
        """ numpy array of UMI counts """
        return np.asarray(self.data, dtype=int)

    @property
    def genes(self):
        """ numpy array of gene names """
        return self.genes

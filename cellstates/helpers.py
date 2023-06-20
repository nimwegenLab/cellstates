import pandas as pd
import numpy as np
from scipy.special import gammaln
from .cluster import Cluster
from .chelpers import marker_scores


# ------ helper functions for cluster hierarchies ------

def get_hierarchy_df(cluster_hierarchy, delta_LL_history):
    """
    create pandas DataFrame from output of get_cluster_hierarchy method
    of Cluster class object.
    """
    hierarchy_df = pd.DataFrame(
        columns=[ 'cluster_new', 'cluster_old', 'delta_LL'],
        index=np.arange( len(cluster_hierarchy)))
    hierarchy_df.loc[:, ['cluster_new', 'cluster_old']] = np.array(cluster_hierarchy)
    hierarchy_df.loc[:, 'delta_LL'] = delta_LL_history

    return hierarchy_df


def hierarchy_to_newick(hierarchy_df, clusters,
                        cell_leaves=True, distance=True, min_distance=0.):
    """
    Function for getting a newick string from a hierarchy DataFrame.

    Parameters
    ----------
    hierachy_df : DataFrame containing cluster merges
    clusters : numpy array, default=None
        initial cluster configuration.
    cell_leaves : bool, default=True
        whether to include cells as leaves; otherwise clusters are leaves
    distance : bool, default=True
        whether to include distance (negative change in log-likelihood)
        in newick tree
    min_distance : float, default=0.
        minimal branch length for very small or positive changes in
        log-likelihood

    Returns
    -------
    newick_string : str
        string of cluster hierarchy in Newick format
    """
    cluster_names = np.unique(clusters)
    cluster_string_dict = {}
    cluster_distance = {c:min_distance for c in cluster_names}
    for c in cluster_names:
        if cell_leaves:
            cluster_args = np.argwhere(clusters==c).flatten().astype(str)
            if distance:
                cluster_string = '(' + (f':{min_distance},').join(cluster_args) + f':{min_distance})C{c}'
            else:
                cluster_string = '(' + ','.join(cluster_args) + f')C{c}'
        else:
            cluster_string = f'C{c}'

        cluster_string_dict[c] = cluster_string

    c_low = min(cluster_string_dict.keys())

    if distance:
        distances = np.cumsum(np.where( (hierarchy_df.delta_LL >= 0) , min_distance, -hierarchy_df.delta_LL + min_distance))

    # use index i instead of step in case hierarchy_df.index is non-standard
    i = hierarchy_df.shape[0] - 1
    for step, row in hierarchy_df.iterrows():
        c_old = row.cluster_old
        c_new = row.cluster_new
        s_old = cluster_string_dict[c_old]
        s_new = cluster_string_dict[c_new]
        if distance:
            d = distances[-i-1]
            d_old = cluster_distance[c_old]
            d_new = cluster_distance[c_new]
            cluster_string_new = f'({s_new}:{d-d_new},{s_old}:{d-d_old})I{i}'
            cluster_distance[c_new] = d
        else:
            cluster_string_new = f'({s_new},{s_old})I{i}'

        cluster_string_dict[c_new] = cluster_string_new
        del cluster_string_dict[c_old]
        del cluster_distance[c_old]

        i-=1

    newick_string = cluster_string_dict[c_low] + ';'
    return newick_string

def get_scipy_hierarchy(hierarchy_df, return_labels=False):
    """
    function to get scipy.cluster.hierarchy linkage matrix

    Parameters
    ----------
    hierachy_df : DataFrame containing cluster merges
    return_labels : whether to return leaf labels

    Returns
    -------
    Z : ndarray
        scipy linkage matrix
    labels : 1D array, optional
        leaf labels that can be used in scipy dendrogram
    """
    N_steps = hierarchy_df.shape[0]
    delta_LL_history = - hierarchy_df.delta_LL.values

    min_delta_LL = np.min(delta_LL_history)
    if min_delta_LL < 0:
        # need to renormalize to only have positive values
        delta_LL_offset = - min_delta_LL
    else:
        delta_LL_offset = 0.

    Z = np.zeros((N_steps, 4))
    Z[:, 2] = delta_LL_history + delta_LL_offset
    cluster_names = np.unique(hierarchy_df.iloc[:, :2]).astype(int)
    clusterindex = dict(zip(cluster_names, range(N_steps+1)))
    clustersize = dict(zip(cluster_names, range(N_steps+1)))
    for i, row in hierarchy_df.iterrows():
        idx_old, idx_new = int(row.cluster_old), int(row.cluster_new)
        cs = clustersize[idx_old] + clustersize[idx_new]
        clustersize[idx_new] = cs
        clustersize[idx_old] = 0
        Z[i, 3] = cs


        Z[i, 0] = min(clusterindex[idx_old], clusterindex[idx_new])
        Z[i, 1] = max(clusterindex[idx_old], clusterindex[idx_new])

        clusterindex[idx_new] = N_steps + 1 + i
        clusterindex[idx_old] = -1

    if return_labels:
        return Z, cluster_names
    else:
        return Z


def clusters_from_hierarchy(hierarchy_df, cluster_init=None, steps=None):
    """
    Get merged clusters from hierarchy_df
    hierachy_df : DataFrame containing cluster merges
    cluster_init : numpy array, default=None
        initial cluster configuration.
        If None, use np.arange(N+1) where N is size of hierarchy
    steps : int, default=-1
        Number of merging steps; if negative perform N+steps steps.
        E.g. with steps=-1 all except the last merge are performed resulting
        in 2 clusters.
    """
    N = hierarchy_df.shape[0]
    if steps < 0:
        steps = N + steps
    clusters = np.arange(N + 1) if cluster_init is None else cluster_init.copy()
    for step in range(steps):
        line = hierarchy_df.iloc[step]
        c_old = line['cluster_old']
        c_new = line['cluster_new']
        clusters[clusters==c_old] = c_new
    return clusters


# ------ functions for finding marker genes ------

def binomial_p(n, lam):
    """

    """
    lam_sum = np.sum(lam)
    n_sum = np.sum(n)
    P =   gammaln(lam_sum) - gammaln(lam) - gammaln(lam_sum - lam) \
        + gammaln(n + lam) + gammaln(n_sum + lam_sum - n - lam) \
        - gammaln(n_sum + lam_sum)

    return P

def gene_contribution(n1, n2, lam):
    d = binomial_p(n1 + n2, lam) - binomial_p(n1, lam) - binomial_p(n2, lam)
    return d

def gene_contribution_multi(all_n, lam):
    d = 0
    all_n_sum = np.zeros_like(all_n[0])
    for n in all_n:
        d -= binomial_p(n, lam)
        all_n_sum += n
    d += binomial_p(all_n_sum, lam)
    return d

def gene_contribution_table(clst, hierarchy_df):
    """
    Returns a table that, for each step in the cluster hierarchy, quantifies
    how much each gene contributes to the change in log-likelihood when these
    clusters are merged. In other words, it gives a score for each gene for
    how different its mean expression is between branches.

    Parameters
    ----------
    clst : cellstate.Cluster object
    hierarchy_df : hierarchy DataFrame of clst

    Returns
    -------
    score_table : (N_merges, N_genes) numpy array of floats
        each row corresponds to a row in hierarchy_df, each column to a gene.
        Values indicate single gene contributions to change in log-likelihood
        of two clusters being merged - large negative values are marker genes
    """
    orignal_clusters = clst.clusters.copy()

    score_table = np.zeros((hierarchy_df.shape[0], clst.G))
    for i, row in hierarchy_df.iterrows():
        c_old, c_new = int(row.cluster_old), int(row.cluster_new)
        d = gene_contribution(clst.cluster_umi_counts[:, c_old],
                              clst.cluster_umi_counts[:, c_new],
                              clst.dirichlet_pseudocounts)
        score_table[i, :] = d

        clst.combine_two_clusters(c_new, c_old)
    clst.set_clusters(orignal_clusters)

    return score_table


def marker_score_table(clst, hierarchy_df):
    """
    Get marker gene scores for each step in a cluster hierarchy.

    Parameters
    ----------
    clst : cellstate.Cluster object
    hierarchy_df : hierarchy DataFrame of clst

    Returns
    -------
    marker_table : (N_merges, N_genes) numpy array of floats
        each row corresponds to a row in hierarchy_df, each column to a gene.
        Values indicate single gene contributions to change in log-likelihood
        of two clusters being merged - large negative values are marker genes
    """
    # create list where element i is a list of cellstates in the cluster
    # initially, only one cellstate is in each cluster, but they will get
    # merged
    cellstate_clusters = []
    for i in range(clst.N_boxes):
        if clst.cluster_sizes[i]:
            cellstate_clusters.append([i])
        else:
            cellstate_clusters.append([])

    score_table = np.zeros((hierarchy_df.shape[0], clst.G))
    for i, row in hierarchy_df.iterrows():
        c_old, c_new = int(row.cluster_old), int(row.cluster_new)
        d = marker_scores(clst, cellstate_clusters[c_new], cellstate_clusters[c_old])
        score_table[i, :] = d

        cellstate_clusters[c_new].extend(cellstate_clusters[c_old])

    return score_table

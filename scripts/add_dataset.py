"""
Python script to add new datasets to a given cellstates partition.
Three parameters can be set inside this script:
    LOG_LEVEL: verbosity of status information; see logging package
    N_CACHE: number of log-gamma values stored for each gene
    TPS: tries per step in MCMC optimization
"""
from cellstates.cluster import Cluster
from cellstates.run import run_mcmc
from cellstates.helpers import get_hierarchy_df, marker_score_table
import numpy as np
import pandas as pd
import argparse
import logging
import os
import time

LOG_LEVEL='INFO'
logformat = '%(asctime)-15s - %(levelname)s:%(message)s'
logging.basicConfig(format=logformat, level=getattr(logging, LOG_LEVEL))

N_CACHE = 10000
TPS = 1000  # tries_per_step in run_mcmc function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O', '--old-data', dest='old_data',
                        default=None, nargs='+',
                        help='UMI data (path to file) with cellstates results',
                        type=str, required=True)
    parser.add_argument('-N', '--new-data', dest='new_data',
                        default=None, nargs='+',
                        help='UMI data (path to file) without cellstates results',
                        type=str, required=True)
    parser.add_argument('-r', '--resultsdir', default='./',
                        help='directory of previous cellstates results', type=str)
    parser.add_argument('-o', '--outdir', default='./',
                        help='directory for new output', type=str)
    parser.add_argument('-g', '--genes', default=None,
                        help='gene names (path to file)', type=str)
    parser.add_argument('--new-genes', default=None, dest='new_genes',
                        help='gene names (path to file) of new data if different', type=str)
    parser.add_argument('-c', '--cells', default=None, nargs='*',
                        help='cell names/barcodes (path to file)', type=str)
    parser.add_argument('-t', '--threads', default=1,
                        help='number of threads', type=int)
    parser.add_argument('-s', '--seed', default=1,
                        help='seed for random generator', type=int)
    parser.add_argument('--save-intermediates', dest='save_intermediates',
                        action='store_true',
                        help='regularly save intermediate results')
    parser.add_argument('--dirichlet-file', default=None, dest='dirichlet_file',
                        help='dirichlet prior parameter file from previous run',
                        type=str)
    parser.set_defaults(optimize_prior=True)
    args = parser.parse_args()

    # -------- load input data -------- #

    # iterate over datafiles and load them
    # data and cellnames are added to lists for concatenating
    all_df = []

    # gene names for old dataset
    if args.genes is not None:
        old_genes = np.loadtxt(args.genes, dtype=str, delimiter='\n')
    else:
        old_genes = None

    # gene names for new dataset
    if args.new_genes is not None:
        new_genes = np.loadtxt(args.new_genes, dtype=str, delimiter='\n')
    elif old_genes is not None:
        # assume genes are the same if not specified otherwise
        new_genes = old_genes
    else:
        new_genes = None

    for datafiles, genes in ((args.old_data, old_genes), (args.new_data, new_genes)):
        index = 0
        for datafile in datafiles:
            logging.info(f'loading {datafile}')

            # check filetype and run appropriate loading function
            filetype=datafile.split('.')[-1]
            filename=datafile.split('.')[:-1]
            if filetype in ['txt', 'tsv', 'zip', 'gz', 'bz2', 'xz', 'csv']:
                df = pd.read_csv(datafile, delim_whitespace=True, header=0, index_col=0)
                if df.shape[1]==1:
                    # if above fails, use slower method and infer delimiter
                    df = pd.read_csv(datafile, sep=None, header=0, index_col=0,
                                    engine='python')
            elif filetype=='npy':
                data = np.load(datafile)
                cells = [f'{filename}-cell_{i}' for i in range(data.shape[1])]
                df = pd.DataFrame(data, index=genes, columns=cells)
            elif filetype=='mtx':
                import scipy.io as sio
                data = sio.mmread(datafile).toarray()
                cells = [f'{filename}-cell_{i}' for i in range(data.shape[1])]
                df = pd.DataFrame(data, index=genes, columns=cells)
            else:
                raise ValueError('filetype not recognized', datafile)

            index += data.shape[1]

            all_df.append(df)

        # remember which datasets were clustered previously
        if datafiles == args.old_data:
            min_index = index

    # build final data table and cell list
    df = pd.concat(all_df, axis=1).fillna(0)

    if args.cells is not None:
        all_cells = []
        for cellfile in args.cells:
            all_cells.append(np.loadtxt(cellfile, dtype=str))
        cells = np.concatenate(all_cells)
    else:
        all_cells = df.columns.values
    genes = df.index.values

    # round to nearest int and cast to np.int64 [long] dtype
    data = df.values
    if np.issubdtype(data.dtype, np.floating):
        data = np.rint(data, out=data)
    data = data.astype(np.int64, copy=False)

    # -------- initialise model parameters -------- #

    # construct initial array of Dirichlet pseudocounts with scale alpha
    if args.dirichlet_file is not None:
        alpha = np.loadtxt(args.dirichlet_file).sum()
    else:
        alpha = np.loadtxt(os.path.join(args.resultsdir,
                                        'dirichlet_pseudocounts.txt')).sum()
    # whether pseudocounts are optimized
    find_best_alpha = args.optimize_prior

    LAMBDA = alpha*np.sum(data, axis=1)/np.sum(data)
    logging.debug(f'using dirichlet prior parameter alpha={alpha}')

    # filter out non-expressed genes
    mask = LAMBDA > 0
    data = data[mask, :]
    LAMBDA = LAMBDA[mask]
    genes = genes[mask]
    G, N = data.shape

    # construct initial cluster array
    if args.init is None:
        cluster_init = np.loadtxt(os.path.join(args.resultsdir,
                                            'optimized_clusters.txt'), dtype=int)
        a = np.max(cluster_init) + 1
        b = N - min_index + a
        cluster_init = np.concatenate([cluster_init, np.arange(a, b, dtype=int)])
    else:
        # assume initialisation is a file from a previous run
        cluster_init = np.loadtxt(args.init, dtype=int)

    clst = Cluster(data, LAMBDA, cluster_init.copy(), max_clusters=b+1,
                   num_threads=args.threads, n_cache=N_CACHE, seed=args.seed)

    # -------- run optimization algorithm -------- #

    if args.save_intermediates:
        dirichlet_file = os.path.join(args.outdir, 'dirichlet_pseudocounts.txt')
        np.savetxt(dirichlet_file, clst.dirichlet_pseudocounts)
        intermediate_dir = args.outdir
    else:
        intermediate_dir = None

    run_mcmc(clst, N_steps=N, log_level=LOG_LEVEL, tries_per_step=TPS,
             results_dir=intermediate_dir, min_index=min_index)

    # -------- Save outputs of model -------- #

    logging.info(f'saving results to directory {args.outdir}')

    logging.debug('save dirichlet pseudocounts used.')
    cluster_file = os.path.join(args.outdir, 'dirichlet_pseudocounts.txt')
    np.savetxt(cluster_file, clst.dirichlet_pseudocounts)

    logging.debug('get cluster hierarchy.')
    hierarchy_file = os.path.join(args.outdir, 'cluster_hierarchy.tsv')
    cluster_hierarchy, delta_LL_history = clst.get_cluster_hierarchy()

    # sometimes clusters still need to be merged
    # find maximum in total LL change after each merge
    total_delta = np.cumsum(delta_LL_history)
    a_max = np.argmax(total_delta)
    # do optimal merges if they increase total LL
    if total_delta[a_max] > 0.:
        for c1, c2 in cluster_hierarchy[:a_max+1]:
            clst.combine_two_clusters(c1, c2)
        cluster_hierarchy = cluster_hierarchy[a_max+1:]
        delta_LL_history = delta_LL_history[a_max+1:]

        # re-map cluster names
        cluster_map = clst.set_N_boxes(clst.n_clusters)
        reverse_map = {v:k for k,v in cluster_map.items()}
        cluster_hierarchy = np.vectorize(reverse_map.__getitem__)(cluster_hierarchy)

    hierarchy_df = get_hierarchy_df(cluster_hierarchy, delta_LL_history)
    logging.debug('save cluster hierarchy as cluster_hierarchy.tsv')
    hierarchy_df.to_csv(hierarchy_file, sep='\t', index=None)

    logging.debug('save cell names')
    cellfile = os.path.join(args.outdir, 'CellID.txt')
    np.savetxt(cellfile, cells, fmt='%s')

    logging.debug('save clusters as optimized_clusters.txt')
    cluster_file = os.path.join(args.outdir, 'optimized_clusters.txt')
    np.savetxt(cluster_file, clst.clusters, fmt='%i')

    logging.debug('get marker gene scores')
    score_table = marker_score_table(clst, hierarchy_df)
    score_df = pd.DataFrame(score_table, columns=genes)
    score_file = os.path.join(args.outdir, 'marker_gene_scores.tsv')
    logging.debug('save marker gene scores as marker_gene_scores.tsv')
    score_df.to_csv(score_file, sep='\t', index=None)


if __name__ == '__main__':
    main()

"""
Python script to run basic cellstates optimization on commandline.
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

LOG_LEVEL='WARNING'
logformat = '%(asctime)-15s - %(levelname)s:%(message)s'
logging.basicConfig(format=logformat, level=getattr(logging, LOG_LEVEL))

N_CACHE = 10000
TPS = 1000  # tries_per_step in run_mcmc function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', default=None, nargs='+',
                        help='UMI data (path to file)', type=str)
    parser.add_argument('-o', '--outdir', default='./',
                        help='directory for output', type=str)
    parser.add_argument('-d', '--dirichlet', default=None,
                        help='dirichlet prior parameter', type=float)
    parser.add_argument('--optimize_dirichlet', action='store_true',
                        help='whether to optimize the dirichlet prior')
    parser.add_argument('-i', '--init', default=None,
                        help='init clusters (path to file)', type=str)
    parser.add_argument('-g', '--genes', default=None,
                        help='gene names (path to file)', type=str)
    parser.add_argument('-c', '--cells', default=None, nargs='*',
                        help='cell names/barcodes (path to file)', type=str)
    parser.add_argument('-t', '--threads', default=1,
                        help='number of threads', type=int)
    parser.add_argument('-s', '--seed', default=1,
                        help='seed for random generator', type=int)

    args = parser.parse_args()

    datafiles = args.data
    all_data = []
    all_cells = []
    genes = None
    for datafile in datafiles:
        logging.info(f'loading {datafile}')
        filetype=datafile.split('.')[-1]
        filename=datafile.split('.')[:-1]

        if filetype in ['txt', 'tsv', 'zip', 'gz', 'bz2', 'xz', 'csv']:
            df = pd.read_csv(datafile, delim_whitespace=True, header=0, index_col=0)
            if df.shape[1]==1:
                # if above fails, use slower method and infer delimiter
                df = pd.read_csv(datafile, sep=None, header=0, index_col=0,
                                engine='python')
            genes = df.index.values
            cells = df.columns.values
            data = df.values
        elif filetype=='npy':
            data = np.load(datafile)
            cells = [f'{filename}-cell_{i}' for i in range(data.shape[1])]
        elif filetype=='mtx':
            import scipy.io as sio
            data = sio.mmread(datafile).toarray()
            cells = [f'{filename}-cell_{i}' for i in range(data.shape[1])]
        else:
            raise ValueError('filetype not recognized', datafile)
        if np.issubdtype(data.dtype, np.floating):
            data = data.rint(data, out=data)
        data = data.astype(np.int64, copy=False)
        all_data.append(data)
        all_cells.append(cells)
    data = np.concatenate(all_data, axis=1)

    if args.cells is not None:
        all_cells = []
        for cellfile in args.cells:
            all_cells.append(np.loadtxt(cellfile, dtype=str))
    cells = np.concatenate(all_cells)

    if args.genes is not None:
        genes = np.loadtxt(args.genes, dtype=str)
    elif genes is None:
        genes = np.arange(data.shape[0], dtype=int)

    if args.dirichlet is None:
        alpha = 2**(np.round(np.log2(data.sum()/data.shape[1])))
    else:
        alpha = args.dirichlet
        if alpha <= 0:
            raise ValueError('dirichlet prior parameter must be > 0')
    find_best_alpha = args.optimize_dirichlet

    logging.info(f'writing to directory {args.outdir}')

    LAMBDA = alpha*np.sum(data, axis=1)/np.sum(data)
    logging.info(f'using dirichlet prior parameter alpha={alpha}')

    # filter out non-expressed genes
    mask = LAMBDA > 0
    data = data[mask, :]
    LAMBDA = LAMBDA[mask]
    genes = genes[mask]
    G, N = data.shape

    if args.init is None:
        cluster_init = np.arange(N, dtype=np.int32)
        logging.info('initialize clusters with all cells seperate')
    elif args.init.endswith('.npy'):
        cluster_init = np.load(args.init)
        logging.info(f'initialize clusters from {args.init}')
    else:
        cluster_init = np.loadtxt(args.init, dtype=str)
        cluster_map = {cn: i for i, cn in enumerate(np.unique(cluster_init))}
        cluster_init = np.array([cluster_map[cn] for cn in cluster_init])
        logging.info(f'initialize clusters from {args.init} with mapping {cluster_map}')

    clst = Cluster(data, LAMBDA, cluster_init.copy(),
                   num_threads=args.threads, n_cache=N_CACHE, seed=args.seed)
    run_mcmc(clst, N_steps=N, log_level=LOG_LEVEL, tries_per_step=TPS)

    # optimise alpha and run MCMC again if needed
    while find_best_alpha:
        best_alpha = alpha
        best_likelihood = clst.total_likelihood

        # check if increasing alpha increases LL
        a = alpha
        logging.info(f'alpha={a}, total_likelihood={clst.total_likelihood}')
        while True:
            a = a*2
            clst.set_dirichlet_pseudocounts(a, n_cache=0)
            logging.info(f'alpha={a}, total_likelihood={clst.total_likelihood}')
            if clst.total_likelihood > best_likelihood:
                best_likelihood = clst.total_likelihood
                best_alpha = a
            else:
                break

        # if not, check if decreasing alpha increases LL
        if best_alpha==alpha:
            a = alpha
            while True:
                a = a/2
                clst.set_dirichlet_pseudocounts(a, n_cache=0)
                logging.info(f'alpha={a}, total_likelihood={clst.total_likelihood}')
                if clst.total_likelihood > best_likelihood:
                    best_likelihood = clst.total_likelihood
                    best_alpha = a
                else:
                    break

        clst.set_dirichlet_pseudocounts(best_alpha, n_cache=N_CACHE)
        logging.info(f'alpha={best_alpha}, total_likelihood={clst.total_likelihood}')
        # if best_alpha is different, run optimization with new value
        if best_alpha!=alpha:
            logging.info(f'run MCMC with new dirichlet prior parameter ' \
                         f'alpha={best_alpha}')
            clst.set_clusters(cluster_init.copy())
            run_mcmc(clst, N_steps=N, log_level=LOG_LEVEL, tries_per_step=TPS)
            alpha = best_alpha
        else:
            logging.info(f'optimal dirichlet prior parameter is ' \
                         f'alpha={best_alpha}')
            find_best_alpha=False

    logging.info('save dirichlet pseudocounts used.')
    cluster_file = os.path.join(args.outdir, 'dirichlet_pseudocounts.txt')
    np.savetxt(cluster_file, clst.dirichlet_pseudocounts)

    logging.info('get cluster hierarchy.')
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
    logging.info('save cluster hierarchy as cluster_hierarchy.tsv')
    hierarchy_df.to_csv(hierarchy_file, sep='\t', index=None)

    logging.info('save cell names')
    cellfile = os.path.join(args.outdir, 'CellID.txt')
    np.savetxt(cellfile, cells, fmt='%s')

    logging.info('save clusters as optimized_clusters.txt')
    cluster_file = os.path.join(args.outdir, 'optimized_clusters.txt')
    np.savetxt(cluster_file, clst.clusters, fmt='%i')

    logging.info('get marker gene scores')
    score_table = marker_score_table(clst, hierarchy_df)
    score_df = pd.concat([hierarchy_df,
                          pd.DataFrame(score_table, columns=genes)],
                         axis=1)
    score_file = os.path.join(args.outdir, 'hierarchy_gene_scores.tsv')
    logging.info('save marker gene scores as hierarchy_gene_scores.tsv')
    score_df.to_csv(score_file, sep='\t', index=None)


if __name__ == '__main__':
    main()

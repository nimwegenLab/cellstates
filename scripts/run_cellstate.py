"""
Python script to run basic cellstates optimization on commandline.
Two parameters can be set inside this script:
    LOG_LEVEL: verbosity of status information; see logging package
    N_CACHE: number of log-gamma values stored for each gene
"""
from cellstates.cluster import Cluster
from cellstates.run import run_mcmc
from cellstates.helpers import get_hierarchy_df, marker_score_table
import numpy as np
import pandas as pd
import argparse
import logging
import os

LOG_LEVEL='INFO'
logformat = '%(asctime)-15s - %(levelname)s:%(message)s'
logging.basicConfig(format=logformat, level=getattr(logging, LOG_LEVEL))

N_CACHE = 10000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', default=None,
                        help='UMI data (path to file)', type=str)
    parser.add_argument('-o', '--outdir', default='./',
                        help='directory for output', type=str)
    parser.add_argument('-d', '--dirichlet', default=None,
                        help='dirichlet prior parameter', type=float)
    parser.add_argument('-i', '--init', default=None,
                        help='init clusters (path to file)', type=str)
    parser.add_argument('-t', '--threads', default=1,
                        help='number of threads', type=int)
    parser.add_argument('-s', '--seed', default=1,
                        help='seed for random generator', type=int)

    args = parser.parse_args()

    datafile = args.data
    filetype=datafile.split('.')[-1]

    if filetype in ['txt', 'tsv', 'zip', 'gz', 'bz2', 'xz']:
        df = pd.read_csv(datafile, sep='\t', header=0, index_col=0)
        df = df.astype(np.int, copy=False)
        genes = df.index.values
        data = df.values
    elif filetype=='npy':
        data = np.load(datafile)
        data = data.astype(np.int, copy=False)
        genes = np.arange(data.shape[0], dtype=int)
    else:
        raise ValueError('filetype not recognized')

    if args.dirichlet is None:
        find_best_alpha=True
        alpha = 2**(np.round(np.log2(data.sum()/data.shape[1])))
    else:
        alpha = args.dirichlet
        if alpha <= 0:
            raise ValueError('dirichlet prior parameter must be > 0')

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
        cluster_init = np.arange(N, dtype=np.int)
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
    run_mcmc(clst, N_steps=N, log_level=LOG_LEVEL)

    # optimise alpha and run MCMC again if needed
    while find_best_alpha:
        best_alpha = alpha
        best_likelihood = clst.total_likelihood

        # check if increasing alpha increases LL
        a = alpha
        while True:
            a = alpha*2
            clst.set_dirichlet_pseudocounts(a, n_cache=0)
            if clst.total_likelihood > best_likelihood:
                best_likelihood = clst.total_likelihood
                best_alpha = a
            else:
                break

        # if not, check if decreasing alpha increases LL
        if best_alpha==alpha:
            a = alpha
            while True:
                a = alpha/2
                clst.set_dirichlet_pseudocounts(a, n_cache=0)
                if clst.total_likelihood > best_likelihood:
                    best_likelihood = clst.total_likelihood
                    best_alpha = a
                else:
                    break

        clst.set_dirichlet_pseudocounts(best_alpha, n_cache=N_CACHE)
        # if best_alpha is different, run optimization with new value
        if best_alpha!=alpha:
            logging.info(f'run MCMC with new dirichlet prior parameter' \
                         f'alpha={best_alpha}')
            run_mcmc(clst, N_steps=N, log_level=LOG_LEVEL)
        else:
            find_best_alpha=False

    logging.info('save dirichlet pseudocounts used.')
    cluster_file = os.path.join(args.outdir, 'dirichlet_pseudocounts.txt')
    np.savetxt(cluster_file, clst.dirichlet_pseudocounts)

    logging.info('save clusters as optimized_clusters.txt')
    cluster_file = os.path.join(args.outdir, 'optimized_clusters.txt')
    np.savetxt(cluster_file, clst.clusters, fmt='%i')

    logging.info('get cluster hierarchy.')
    hierarchy_file = os.path.join(args.outdir, 'cluster_hierarchy.tsv')
    cluster_hierarchy, delta_LL_history = clst.get_cluster_hierarchy()

    hierarchy_df = get_hierarchy_df(cluster_hierarchy, delta_LL_history)
    logging.info('save cluster hierarchy as cluster_hierarchy.tsv')
    hierarchy_df.to_csv(hierarchy_file, sep='\t', index=None)

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

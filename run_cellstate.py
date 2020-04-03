from cellstates.cluster import Cluster
from cellstates.run import run_mcmc
from cellstates.helpers import get_hierarchy_df
import numpy as np
import pandas as pd
import argparse
import logging
import os

LOG_LEVEL='INFO'
logformat = '%(asctime)-15s - %(levelname)s:%(message)s'
logging.basicConfig(format=logformat, level=getattr(logging, LOG_LEVEL))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', default=None,
                        help='directory for output', type=str)
    parser.add_argument('--data', default=None,
                        help='UMI data (path to file)', type=str)
    parser.add_argument('-d', '--dirichlet', default=None,
                        help='dirichlet prior parameter', type=float)
    parser.add_argument('-i', '--init', default=None,
                        help='init clusters (path to file)', type=str)
    parser.add_argument('-t', '--threads', default=1,
                        help='number of threads', type=int)
    args = parser.parse_args()

    datafile = args.data
    if datafile.endswith('.txt') or datafile.endswith('.tsv'):
        df = pd.read_csv(datafile, sep='\t', header=0, index_col=0)
        df = df.astype(np.int, copy=False)
        data = df.values
    elif datafile.endswith('.npy'):
        data = np.load(datafile)
        data = data.astype(np.int, copy=False)

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
                   num_threads=args.threads, n_cache=10000)
    run_mcmc(clst, N_steps=N, log_level=LOG_LEVEL)

    # optimise alpha and run MCMC again if needed
    while find_best_alpha:
        best_alpha = alpha
        best_likelihood = clst.total_likelihood
        max_clusters = np.max(clst.clusters)+1
        clusters = clst.clusters.copy()

        # check if increasing alpha increases LL
        a = alpha
        while True:
            a = alpha*2
            clst_trial = Cluster(data, a*LAMBDA/alpha, clusters,
                                 max_clusters=max_clusters, n_cache=0)
            if clst_trial.total_likelihood > best_likelihood:
                best_likelihood = clst_trial.total_likelihood
                best_alpha = a
            else:
                break

        # if not, check if decreasing alpha increases LL
        if best_alpha==alpha:
            a = alpha
            while True:
                a = alpha/2
                clst_trial = Cluster(data, a*LAMBDA/alpha, clusters,
                                     max_clusters=max_clusters, n_cache=0)
                if clst_trial.total_likelihood > best_likelihood:
                    best_likelihood = clst_trial.total_likelihood
                    best_alpha = a
                else:
                    break

        # if best_alpha is different, run optimization with new value
        if best_alpha!=alpha:
            logging.info(f'run MCMC with new dirichlet prior parameter' \
                         f'alpha={best_alpha}')
            LAMBDA = best_alpha*LAMBDA/alpha
            alpha=best_alpha
            clst = Cluster(data, LAMBDA, cluster_init.copy(),
                        num_threads=args.threads, n_cache=10000)
            run_mcmc(clst, N_steps=N, log_level=LOG_LEVEL)
        else:
            find_best_alpha=False


    logging.info('save clusters as optimized_clusters.txt')
    cluster_file = os.path.join(args.outdir, 'optimized_clusters.txt')
    np.savetxt(cluster_file, clst.clusters, fmt='%i')

    logging.info('get cluster hierarchy.')
    hierarchy_file = os.path.join(args.outdir, 'cluster_hierarchy.tsv')
    cluster_hierarchy, delta_LL_history = clst.get_cluster_hierarchy()

    hierarchy_df = get_hierarchy_df(cluster_hierarchy, delta_LL_history)
    logging.info('save cluster hierarchy as cluster_hierarchy.tsv')
    hierarchy_df.to_csv(hierarchy_file, sep='\t', index=None)


if __name__ == '__main__':
    main()

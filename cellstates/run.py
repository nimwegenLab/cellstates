from .cluster import Cluster
import numpy as np
import pandas as pd
import logging
import os


def run_mcmc(clst, results_dir=None, N_steps=10000, tries_per_step=1000, min_index=0,
             log_level='INFO', keep_intermediate=False):
    """
    function to run full Markov-chain Monte Carlo optimization algorithm on a
    Cluster object and save outputs in files.

    Parameters
    ----------
    clst : Cluster object
    results_dir : str, default=None
        path to directory where the latest intermediate cluster state is stored
        as clusters_intermediate.txt at each breakpoint.
        If None or empty string, nothing is saved.
    N_steps : int, default=10000
        Number of MCMC steps to initially run per breakpoint.
    tries_per_step : int, default=1000
        Number of moves initially proposed per step. The higher, the longer
        the optimization can last (and the better the optimum)
    log_level : {'DEBUG', 'INFO', 'WARNING', 'ERROR'}, default='INFO'
        verbosity of information of progression. Set to 'ERROR' to turn off.
    keep_intermediate : bool, default=False
        all intermediate cluster states are saved as
        clusters_****.txt at each breakpoint (**** represents a 4-digit count).

    """
    logformat = '%(asctime)-15s - %(levelname)s:%(message)s'
    logging.basicConfig(format=logformat, level=getattr(logging, log_level))

    logging.debug(f'initially check output every {N_steps} steps')

    i = 0
    # save initial configuration
    if results_dir:
        cluster_file = os.path.join(results_dir, 'intermediate_clusters.txt')
        logging.debug(f'writing intermediate states to directory {results_dir}')
        logging.debug(f'write output {i:04d}')
        if keep_intermediate:
            cluster_file = os.path.join(results_dir, f'clusters_{i:04d}.txt')
        np.savetxt(cluster_file, clst.clusters, fmt='%i')
    logging.debug(f'n_clusters={clst.n_clusters}, ' \
                 f'total likelihood={clst.total_likelihood}')
    old_likelihood = clst.total_likelihood
    best_clusters = clst.clusters.copy()

    while i < 10000:
        i += 1
        if clst.N_boxes != clst.n_clusters + 2:
            clst.set_N_boxes(clst.n_clusters + 2)
            logging.debug(f'changed N_boxes to {clst.N_boxes}')
        try:
            _ = clst.biased_monte_carlo_sampling(N_steps=N_steps,
                                                 tries_per_step=tries_per_step,
                                                 min_index=min_index)
        except RuntimeError as err:
            #N_batch = N_batch*10
            tries_per_step *= 10
            N_steps = N_steps//10
            logging.debug(err)
            if N_steps < 10:
                logging.debug('MCMC converges, little further improvement expected')
                break
            else:
                logging.debug(f'changed tries_per_step to {tries_per_step} and N_steps to {N_steps}')
        finally:
            if clst.total_likelihood > old_likelihood:
                old_likelihood = clst.total_likelihood
                if results_dir:
                    logging.debug(f'write output {i:04d}')
                    if keep_intermediate:
                        cluster_file = os.path.join(results_dir, f'clusters_{i:04d}.txt')
                    np.savetxt(cluster_file, clst.clusters, fmt='%i')
                best_clusters = clst.clusters.copy()
                logging.debug(f'n_clusters={clst.n_clusters}, ' \
                            f'total likelihood={clst.total_likelihood}')

            else:
                logging.debug('likelihood did not improve; clustering converged')
                # revert to better clustering

                clst.set_clusters(best_clusters)
                break

    logging.debug('optimize clusters.')
    clst.optimize_clusters()
    clst.set_N_boxes(clst.n_clusters)
    logging.debug('end of clustering')

# cellstates

`cellstates` is a python package for analysis of UMI-based single-cell RNA-seq data. The underlying mathematical model infers clusters of cells that are in the same gene expression state, meaning that all remaining heterogeneity within each cluster can be explained by expected measurement noise. Thus, we find the most fine-grained clustering that is  supported by the data. Furthermore, we describe the higher-order relationship of these cell-states in a hierarchical tree and provide scores for marker-genes within this tree.

# Installation

Installation of the cellstate python package can simply be done by going in the package folder and running 

If you want to use cellstates from within the folder (for example to use run_cellstates.py), run
`python setup.py build_ext --inplace`
If you want to import the cellstates python module elsewhere, run
`python setup.py build_ext`

Finally, install with
`python setup.py install`

## Requirements
A C-compiler with **OpenMP** is required for installation. See below for possible solutions with MacOS. 

**Python packages**
Mandatory: numpy, pandas, matplotlib, scipy
Strongly recommended: Cython
Recommended for plotting of hierarchies: [ete3](http://etetoolkit.org/)

## MacOS
The default compiler on MacOS does not include OpenMP support. To solve this, we suggest two possible solutions:
### Use conda compilers and environment
Modified solution from [here](https://github.com/scikit-learn/scikit-learn/blob/master/doc/developers/advanced_installation.rst#macos-compilers-from-conda-forge) by creating a conda environment:

```
conda create -n cellstates-env python numpy scipy cython pandas matplotlib \
    "conda-forge::compilers>=1.0.4" conda-forge::llvm-openmp
conda activate cellstates-env
python setup.py build_ext --inplace
```

### Using gcc9
Using Macports:
```
port install gcc9
CC=g++-mp-9 python setup.py build_ext --inplace
```
Using Homebrew:
```
brew install gcc@9
CC=g++-9 python setup.py build_ext --inplace
```




# Commandline tool
The most **basic version**, can be run through the command line as follows:

`python run_cellstate.py --data data.tsv`

The **input** is a table of integer UMI counts. The commandline tool currently supports the following formats:
* A tab-separated values file with .tsv or .txt ending. Columns are cells, rows are genes. 
* A numpy array of integers saved as .npy file


It returns the following **results**:
* `optimized_clusters.txt`: indicates for each cell in which cell-state they are (cellstates are assigned an arbitrary number)
* `cluster_hierarchy.tsv`: The relationship between cell-states can be described through a hierarchical tree. Leaves in this tree are cell-states which are iteratively merged into higher-order clusters. This tree structure is saved as a tab separated value file with three columns:
    * cluster_new: Cluster/cell-state to be merged and label of new merged cluster
    * cluster_old: Other cluster/cell-state to be merged with cluster_new
    * Delta_LL: Change in log-likelihood (usually negative; the more negative, the more different are the merged clusters)
* `hierarchy_gene_scores.tsv`: For each merging-step in the hierarchical tree, we can give a score for how much a gene contributes to the separation of the two branches (large negative score). In this tab-separated value file, the first three columns are the same as in `cluster_hierarchy.tsv`, indicating the merging step, and the following columns are the scores for each gene. 
* `dirichlet_pseudocounts.txt`: The prior parameters for which the optimum was found. 


## Advanced Commandline tool
```
usage: run_cellstate.py [-h] [-o OUTDIR] [-d DIRICHLET] [-i INIT] [-t THREADS]
                        [-s SEED]
                        data

positional arguments:
  data                  UMI data (path to file)

optional arguments:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
                        directory for output
  -d DIRICHLET, --dirichlet DIRICHLET
                        dirichlet prior parameter
  -i INIT, --init INIT  init clusters (path to file)
  -t THREADS, --threads THREADS
                        number of threads
  -s SEED, --seed SEED  seed for random generator
```
Additional comments for selected parameters:
* `DIRICHLET`: If given, the model is run only with the given parameter, otherwise the parameter will be optimized
* `INIT`: Cluster labels should be given in a simple text file separated by line breaks or as a binary .npy file. 
* `THREADS`: Default is one core

# Python module and interpretation of results

Check out the cellstate_introduction.ipynb and Example_analysis.ipynb jupyter notebooks for information about how to use the cellstates python module and how to analyse and interpret outputs.

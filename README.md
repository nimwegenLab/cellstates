# cellstates

`cellstates` is a python package for analysis of UMI-based single-cell RNA-seq data. The underlying mathematical model infers clusters of cells that are in the same gene expression state, meaning that all remaining heterogeneity within each cluster can be explained by expected measurement noise. Thus, we find the most fine-grained clustering that is  supported by the data. Furthermore, we describe the higher-order relationship of these cell-states in a hierarchical tree and provide scores for marker-genes within this tree.

`cellstates` was developed by Pascal Grobecker and Erik van Nimwegen. 

## Installation

If you want to use cellstates from within the folder, run

`python setup.py build_ext --inplace`

If you want to import the cellstates python module elsewhere, run

`python setup.py build_ext`

Finally, install with

`python setup.py install`

### Requirements
A C-compiler with OpenMP is required for installation. See below for possible solutions with MacOS. 

**Python packages**
* Mandatory: numpy, pandas, matplotlib, scipy
* Strongly recommended: Cython
* Recommended for plotting of hierarchies: [ete3](http://etetoolkit.org/)

### MacOS
The default compiler on MacOS does not include OpenMP support. To solve this, we suggest two possible solutions:
#### Use conda compilers and environment
Modified solution from [here](https://github.com/scikit-learn/scikit-learn/blob/master/doc/developers/advanced_installation.rst#macos-compilers-from-conda-forge) by creating a conda environment:

```
conda create -n cellstates-env python numpy scipy cython pandas matplotlib \
    "conda-forge::compilers>=1.0.4" conda-forge::llvm-openmp
conda activate cellstates-env
python setup.py build_ext --inplace
```

#### Using gcc9
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




## Commandline tool
### Basic instructions
The most basic version, can be run through the command line as follows:

`python ./scripts/run_cellstates.py data.tsv`

The **input** is a table of integer UMI counts. 
If multiple files are given, the data tables will be concatenated under the assumption that all rows correspond to the same genes.
The commandline tool currently supports the following formats:
* A tab-separated values file with .tsv or .txt ending. Columns are cells, rows are genes. CSV-files (.csv) or compressed files recognized by pandas read\_csv method should work too. 
* A numpy array of integers saved as .npy file
* Matrix Market .mtx file


It returns the following **results**:
* `optimized_clusters.txt`: indicates for each cell in which cell-state they are (cellstates are assigned an arbitrary number)
* `cluster_hierarchy.tsv`: The relationship between cell-states can be described through a hierarchical tree. Leaves in this tree are cell-states which are iteratively merged into higher-order clusters. This tree structure is saved as a tab separated value file with three columns:
    * cluster\_new: Cluster/cell-state to be merged and label of new merged cluster
    * cluster\_old: Other cluster/cell-state to be merged with cluster\_new
    * Delta\_LL: Change in log-likelihood (usually negative; the more negative, the more different are the merged clusters)
* `hierarchy_gene_scores.tsv`: For each merging-step in the hierarchical tree, we can give a score for how much a gene contributes to the separation of the two branches (large negative score). In this tab-separated value file, the first three columns are the same as in `cluster_hierarchy.tsv`, indicating the merging step, and the following columns are the scores for each gene. 
* `dirichlet_pseudocounts.txt`: The prior parameters for which the optimum was found. 
* `CellID.txt` list of cell names/barcodes of the concatenated data table


### Advanced instructions
```
usage: run_cellstates.py [-h] [-o OUTDIR] [-d DIRICHLET]
                         [--prior-optimization] [--no-prior-optimization]
                         [-i INIT] [-g GENES] [-c [CELLS [CELLS ...]]]
                         [-t THREADS] [-s SEED]
                         data [data ...]

positional arguments:
  data                  UMI data (path to file)

optional arguments:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
                        directory for output
  -d DIRICHLET, --dirichlet DIRICHLET
                        dirichlet prior parameter
  --prior-optimization  add flag to optimize the dirichlet prior
                        [Default=True]
  --no-prior-optimization
                        add flag to not optimize the dirichlet prior
  -i INIT, --init INIT  init clusters (path to file)
  -g GENES, --genes GENES
                        gene names (path to file)
  -c [CELLS [CELLS ...]], --cells [CELLS [CELLS ...]]
                        cell names/barcodes (path to file)
  -t THREADS, --threads THREADS
                        number of threads
  -s SEED, --seed SEED  seed for random generator
  --save-intermediates  regularly save intermediate results
  --dirichlet-file DIRICHLET_FILE 
                        dirichlet prior parameter file from previous run
```
Additional comments for selected parameters:
* `INIT`: Cluster labels should be given in a simple text file separated by line breaks or as a binary .npy file. 
* `GENES`: list of gene names should be given in a simple text file separated by line breaks.
* `CELLS`: list of cell names/barcodes should be given in a simple text file separated by line breaks. Multiple files can be given if there are multiple data files.
* `THREADS`: Default is one core

### save and load intermediate progress
`cellstates` can take a long time to run and you may want to save its progress. This can be done as follows:

`python run_cellstates data.tsv --save-intermediates --outdir ./my_results`

If the run is interrupted, it can be resumed by calling

`python run_cellstates data.tsv --init ./my_results/intermediate_clusters.txt --dirichlet-file ./my_results/dirichlet_pseudocounts.txt --outdir ./my_results`

Note that the shown runtime estimate currently does not take previous progress into account. 

## Python module and interpretation of results

Check out the cellstate\_introduction.ipynb and Example_analysis.ipynb jupyter notebooks for information about how to use the cellstates python module and how to analyse and interpret outputs.

## Testing

For now, please check cellstate\_introduction.ipynb for some tests you can run. 

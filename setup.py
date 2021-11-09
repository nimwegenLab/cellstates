from setuptools import setup, Extension
import numpy

NAME = 'cellstates'
VERSION = '0.1'
DESCR = 'Module for finding gene expression states in scRNAseq data'
REQUIRES = ['numpy', 'pandas', 'matplotlib']

AUTHOR = 'Pascal Grobecker'
EMAIL = 'pascal.grobecker@unibas.ch'

PACKAGES = ['cellstates']


USE_CYTHON = True
try:
    from Cython.Build import cythonize
except ModuleNotFoundError:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

EXTENSIONS = [Extension("cellstates.cluster",
                        ["cellstates/cluster" + ext],
                        include_dirs=[numpy.get_include(), '.'],
                        extra_compile_args=['-fopenmp'],
                        extra_link_args=['-fopenmp'],
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
                        )
              ]

if USE_CYTHON:
    EXTENSIONS = cythonize(EXTENSIONS,
                           compiler_directives={'language_level': 3})

if __name__ == '__main__':
    setup(
        name=NAME,
        version=VERSION,
        description=DESCR,
        author=AUTHOR,
        author_email=EMAIL,
        install_requires=REQUIRES,
        packages=PACKAGES,
        ext_modules=EXTENSIONS
    )

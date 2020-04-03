from setuptools import setup
from Cython.Distutils import build_ext, Extension
import cython_gsl
import numpy

NAME = 'cellstates'
VERSION = '0.1'
DESCR = 'Module for finding gene expression states in scRNAseq data'
REQUIRES = ['numpy', 'cython', 'CythonGSL']

AUTHOR = 'Pascal Grobecker'
EMAIL = 'pascal.grobecker@unibas.ch'

PACKAGES = ['cellstates']

ext_1 = Extension("cellstates.cluster",
                  ["cellstates/cluster.pyx"],
                  libraries=cython_gsl.get_libraries(),
                  library_dirs=[cython_gsl.get_library_dir()],
                  include_dirs=[cython_gsl.get_cython_include_dir(),
                                numpy.get_include(), '.'],
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-fopenmp']
                  )

EXTENSIONS = [ext_1]

if __name__ == '__main__':
    setup(
        name=NAME,
        version=VERSION,
        description=DESCR,
        author=AUTHOR,
        author_email=EMAIL,
        install_requires=REQUIRES,
        packages=PACKAGES,
        include_dirs=[cython_gsl.get_include()],
        cmdclass={'build_ext': build_ext},
        ext_modules=EXTENSIONS
    )

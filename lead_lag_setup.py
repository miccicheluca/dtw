from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy as np

# First create an Extension object with the appropriate name and sources.
ext = Extension(
    name="ll_computation",
    sources=["lead_lag_computations_c.pyx"],
    include_dirs=[np.get_include()],
    language="c++",
)
# Use cythonize on the extension object.
setup(ext_modules=cythonize(ext))

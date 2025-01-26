from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("boundvor/geometry.pyx"),
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    packages=["boundvor"],
    package_data={
    "boundvor": ["geometry.pyx"],  # This includes your .pyx file
},
    extras_require={"dev": ["pytest", "cython"]},
)

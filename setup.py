from setuptools import setup, Extension
import numpy

def get_numpy_include_dirs():
    return [numpy.get_include()]

geometry_c_extension = Extension(
    "boundvor.geometry_c",
    sources=["src/geometry.c"],
    include_dirs=get_numpy_include_dirs()
)

setup(
    ext_modules=[geometry_c_extension],
)

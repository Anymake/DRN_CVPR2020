from setuptools import setup, Extension
import numpy as np

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        'pycocotools_ro._mask',
        sources=['../common/maskApi.c', 'pycocotools_ro/_mask.pyx'],
        include_dirs = [np.get_include(), '../common'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99', "-fopenmp"],
    )
]

setup(
    name='pycocotools_ro',
    packages=['pycocotools_ro'],
    package_dir = {'pycocotools_ro': 'pycocotools_ro'},
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib>=2.1.0'
    ],
    version='1.0',
    ext_modules= ext_modules
)

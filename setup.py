from setuptools import find_packages, setup

setup(
    name = "MLGLUE",
    author = "M. G. Rudolph",
    version = "0.0.6",
    description = "a Python implementation of the (M)ulti(l)evel (G)eneralized (L)ikelihood (U)ncertainty (E)stimation (MLGLUE) algorithm and utility functions",
    url = "https://github.com/iGW-TU-Dresden/MLGLUE",
    author_email = "max_gustav.rudolph@tu-dresden.de",
    license = "MIT",
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Other Audience",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Hydrology"
        ],
    platforms = "Windows",
    install_requires = [
        "numpy>=1.23.5",
        "matplotlib>=3.6.3",
        "ray>=2.2.0"
        ],
    packages = find_packages(exclude=[])
    )

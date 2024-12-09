Getting started with MLGLUE
===========================
On this page you will find all the information to get started with MLGLUE.
It is assumed that you are somewhat familiar with programming in Python and
that you have a working Python distribution installed.

Installing MLGLUE
-----------------
The easiest way to install MLGLUE is to use the Python Package Index
(`PyPI <https://pypi.python.org/pypi/mlglue>`_). To get the latest version
of MLGLUE, open the Anaconda Prompt, a Windows Command Prompt (also called
command window) or a Mac/Linux terminal and type::

    pip install mlglue

Updating MLGLUE
---------------
If you have already installed MLGLUE, it is possible to update MLGLUE
easily. To update, open a Anaconda/Windows command prompt or a Mac terminal
and type::

    pip install mlglue --upgrade

Dependencies
------------
MLGLUE depends on a number of Python packages. The necessary packages are
all automatically installed when using pip. The following packages are
necessary for a minimal function installation of MLGLUE::

    numpy
    Ray

Strictly speaking, Ray is not necessary but installing it is very strongly
encouraged (which is why it is listed as a necessary package). Ray enables
the parallel execution of models, often resulting in substantial
computational savings. And due to Monte Carlo-type algorithms (and
therefore MLGLUE) being embarassingly parallel, using Ray almost always
results in substantial speed-up. However, there is Python-native support
included, such that Ray is installed but optionally not used.
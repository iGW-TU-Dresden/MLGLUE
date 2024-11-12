[![image](https://img.shields.io/pypi/v/mlglue.svg)](https://pypi.python.org/pypi/mlglue)
[![image](https://img.shields.io/pypi/l/mlglue.svg)](https://mit-license.org/)
[![image](https://img.shields.io/pypi/pyversions/mlglue)](https://pypi.python.org/pypi/mlglue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13122535.svg)](https://doi.org/10.5281/zenodo.13122535)
[![Documentation Status](https://readthedocs.org/projects/mlglue/badge/?version=latest)](https://mlglue.readthedocs.io/en/latest/?badge=latest)

# MLGLUE
This Python package is an implementation of the Multilevel Generalized Likelihood Uncertainty Estimation (MLGLUE) algorithm, including some utility functions. See the MLGLUE paper [here](https://essopenarchive.org/users/677423/articles/853135-extending-glue-with-multilevel-methods-to-accelerate-statistical-inversion-of-hydrological-models) (preprint).

## Installation
`pip install MLGLUE`

`MLGLUE` uses `Ray` for parallelization. Installation using `pip install MLGLUE` also installs `Ray` as dependency along `numpy` and `matplotlib`. Using a custom installation, please prepare an environment for `Ray` first as described [here](https://docs.ray.io/en/releases-2.2.0/ray-overview/installation.html) for version `2.2.0`. The use of other versions of `Ray` has not been tested.

## Documentation

A documentation webpage is available [here](https://mlglue.readthedocs.io/en/latest/) or alternatively under `mlglue.readthedocs.io`.

## Usage
`MLGLUE` requires the computational model to be given in the form of a callable with a specific set of arguments and returns:

```python
def my_model(parameters, level, n_levels, obs_x, obs_y, likelihood, run_id)
        '''
        Parameters
        ----------
        parameters : 1D list-like
                the model parameters
        level : int
                the 0-based level index
        n_levels : int
                the total number of levels in the hierarchy
        obs_x : 1D list-like
                virtual observation ordinates (not actually used in
                computations)
        obs_y : 1D list-like
                the observations with which to compute the
                likelihood
        likelihood : MLGLUE.Likelihoods instance
        	an instance of any likelihood from MLGLUE.Likelihoods
                with which to compute the likelihood
        run_id : int or str
                an identifier for the model run

        Returns
        -------
        computed_likelihood : float
                the computed likelihood
        simulated_observation_equivalents : 1D list-like
                the simulated equivalents of the observations
        '''

        # your model code
        # using the function parameter_to_observable_map as a placeholder
        #	for more complex models
        simulated_observation_equivalents = parameter_to_observable_map(
        	parameters
        	)

        # compute likelihood
        computed_likelihood = likelihood.likelihood(
        	obs=obs_y,
        	sim=simulated_observation_equivalents
        	)

        return computed_likelihood, simulated_observation_equivalents
```

See the examples directory for a working implementation.
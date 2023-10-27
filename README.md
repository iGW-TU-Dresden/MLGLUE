# MLGLUE
This Python package is an implementation of the Multilevel Generalized Likelihood Uncertainty Estimation (MLGLUE) algorithm, including some utility functions. See the MLGLUE paper [here](doi.org/10.22541/essoar.169833433.35092350/v1) (preprint).

## Installation
`pip install MLGLUE`

`MLGLUE` uses `Ray` for parallelization. To ensure functionality, please prepare an environment for `Ray` first as described [here](https://docs.ray.io/en/releases-2.2.0/ray-overview/installation.html) for version `2.2.0`. The use of other versions of `Ray` has not been tested.

## Usage
`MLGLUE` requires the computational model to be given in the form of a function with a specific set of arguments and returns:

```python
def my_model(parameters, level, n_levels, obs_x, obs_y, likelihood, run_id)
        """
        Parameters
        ----------
        # :param parameters: 1D list-like of model parameters
        # :param level: int representing the 0-based level index
        # :param n_levels: int representing the total number of levels
        # :param obs_x: 1D list-like of virtual observation ordinates
        	(not actually used in computations)
        # :param obs_y: 1D list-like of observations
        # :param likelihood: an instance of an MLGLUE.Likelihoods
        	likelihood instance
        # :param run_id: an int or str identifier for the model run

        Returns
        -------
        :return computed_likelihood: the computed likelihood as float
        :return simulated_observation_equivalents: the simulated
        	equivalents of the observations
        """

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
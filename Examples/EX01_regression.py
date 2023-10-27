import numpy as np
import matplotlib.pyplot as plt
import MLGLUE
import pandas as pd
import time

"""
This example considers a sinple linear regression problem.
Noisy data is first generated from the true process y = m*x + b + Gaussian(0, 0.8).
The number of observations (y-values) is set to 500. Three levels of accuracy are used.
The highest level model considers all 500 observations, the next lower level
considers 250 observations (i.e., every second observation), and the lowest level
considers 125 observations (i.e., every fourth observation). Inferences are made with
respect to the highest level model, however mostly relying on samples from the
lower level models. While this example is computationally very efficient (i.e., a
single model call is very cheap), computational savings with MLGLUE take effect if
a single model call becomes more costly.
See the MLGLUE paper for more details.
"""

def my_model(parameters, level, n_levels, obs_x, obs_y, likelihood, run_id):
    """
    The model for which parameters should be estimated with MLGLUE.
    This structure is mandatory for MLGLUE; adapt the code to your needs; the
    function, however, should always return a likelihood value and return the
    results when on the finest level

    :param parameters: the model parameters; list-like
    :param level: the level index; int
    :param n_levels: the total number of levels; int
    :param obs_x: the observation locations; list-like
    :param obs_y: the observations; list-like
    :param likelihood: the likelihood function; MLGLUE.Likelihoods-function
    :param run_id: an int representing some additional run identifier; int
    :return: likelihood (the float value for the likelihood, is always
        returned), results (the model results, only returned when on the
        highest level)
    """

    try:
        if level >= n_levels:
            msg = ("There is no level {} with n_levels = {}".format(level,
                                                                    n_levels))
            raise ValueError(msg)
    except ValueError:
        raise

    coarsening = 2
    stepper = [int(coarsening ** i) for i in range(1, n_levels)]
    stepper.reverse()

    # handle finest level
    if level == n_levels - 1:
        x_ = obs_x
        y_ = obs_y

        """ make the model artificially slower """
        # time.sleep(.1)
    # handle coarser levels
    else:
        x_ = obs_x[::stepper[level]]
        y_ = obs_y[::stepper[level]]

    y_model = parameters[0] * x_ + parameters[1]

    if y_model is None:
        return None, None

    else:
        # calculate likelihood
        likelihood_ = likelihood.likelihood(obs=y_, sim=y_model)

        return likelihood_, y_model

# generate observations
np.random.seed(5)

true_intercept = 1.
true_slope = 2.
sigma = .8

size=500
x = np.linspace(0, 1, size)
y = true_intercept + true_slope * x + np.random.normal(0, sigma**2, size)

fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
ax.plot(x, y, "x", label="sampled data")
ax.plot(x, true_intercept + true_slope * x, label="true regression line")
plt.legend(loc="best")
# plt.show()

# define likelihood
mylike = MLGLUE.InverseErrorVarianceLikelihood(
    threshold=0.002, T=1.
)

# set random seed
np.random.seed(10)

# start MLGLUE
start = time.time()
mlglue = MLGLUE.MLGLUE(likelihood=mylike, model=my_model,
                       upper_bounds=[4., 2.], lower_bounds=[-4., -2.],
                       obs_x=x, obs_y=y, n_samples=100000, n_levels=3,
                       multiprocessing=True, n_processors=5, tuning=0.02,
                       variance_analysis=None, savefigs="regression")

samples, liks, results = mlglue.perform_MLGLUE()

print("time: ", time.time() - start)

print("Shape of samples: ", np.shape(samples))
print("Shape of results: ", np.shape(results))
print("Shape of likelihoods: ", np.shape(liks))

uncertainty_estimates = mlglue.estimate_uncertainty()
print(uncertainty_estimates)

# save outputs
pd.DataFrame(uncertainty_estimates).to_csv("EX01_regression_uncertainty_estimates.csv")
pd.DataFrame(samples).to_csv("EX01_regression_parameter_samples.csv")

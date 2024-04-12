import numpy as np

class InverseErrorVarianceLikelihood():
    def __init__(self, threshold=0.1, T=2., weights=None):
        """
        :param threshold: the threshold to use, given as a fraction of
            simulations that are accepted (the actual likelihood value for
            the threshold is inferred during the tuning phase of MLGLUE);
            float between 0 and 1
        :param T: a shape parameter of the likelihood function, representing
            more "stricness" with increasing values; float
        :param weights: a list-like of weights for the individual observations
            (if None, all weights are set to 1.) where the length needs
            to be equal to the number of observations, list-like of floats
            or None
        """

        self.threshold = threshold
        self.T = T
        self.weights = weights

        return

    def likelihood(self, obs=None, sim=None):
        """
        Compute the likelihood according to Beven & Binley (1991), i.e., the
        inverse error variance likelihood

        :param obs: a 1D list-like corresponding to the observation y-values;
            list-like
        :param sim: a 1D list-like corresponding to the simulation y-values;
            list-like
        
        :return likelihood: a float value for the likelihood
        """

        try:
            if obs is None and sim is None:
                msg = ("No observations and no simulated values are given!")
                raise ValueError(msg)
            elif obs is None and sim is not None:
                msg = ("No observations are given!")
                raise ValueError(msg)
            elif obs is not None and sim is None:
                msg = ("No simulated values are given!")
                raise ValueError(msg)
        except ValueError:
            raise

        try:
            if len(sim) != len(obs):
                msg = ("Length mismatch! Observed values have length {} but "
                       " simulated values have length {}".format(len(obs),
                                                                 len(sim)))
                return 0.
        except ValueError:
            raise

        if self.weights is None:
            weights = np.ones(len(obs))
        else:
            weights = self.weights

        try:
            if len(weights) != len(obs):
                msg = ("Length mismatch! Weights values have length {} but "
                       " observed values have length {}".format(
                            len(weights), len(obs))
                       )
                raise ValueError(msg)

        except ValueError:
            raise

        # calculate the likelihood
        residuals = np.asarray(obs) - np.asarray(sim)
        residuals *= weights
        ssr = np.sum(residuals ** 2)
        likelihood = (ssr / (len(obs) - 2)) ** (-self.T)

        if np.isinf(likelihood):
            if np.isneginf(likelihood):
                return -1e10
            else:
                return 1e10

        return likelihood

class RelativeVarianceLikelihood():
    def __init__(self, threshold=0.1, weights=None):
        """
        :param threshold: the threshold to use, given as a fraction of
            simulations that are accepted (the actual likelihood value for
            the threshold is inferred during the tuning phase of MLGLUE);
            float between 0 and 1
        :param weights: a list-like of weights for the individual observations
            (if None, all weights are set to 1.) where the length needs
            to be equal to the number of observations, list-like of floats
            or None
        """

        self.threshold = threshold
        self.weights = weights

        return

    def likelihood(self, obs=None, sim=None):
        """
        Compute the model efficiency, i.e., the relative variance likelihood

        :param obs: a 1D list-like corresponding to the observation y-values;
            list-like
        :param sim: a 1D list-like corresponding to the simulation y-values;
            list-like
        
        :return likelihood: a float value for the likelihood
        """

        try:
            if obs is None and sim is None:
                msg = ("No observations and no simulated values are given!")
                raise ValueError(msg)
            elif obs is None and sim is not None:
                msg = ("No observations are given!")
                raise ValueError(msg)
            elif obs is not None and sim is None:
                msg = ("No simulated values are given!")
                raise ValueError(msg)
        except ValueError:
            raise

        try:
            if len(sim) != len(obs):
                msg = ("Length mismatch! Observed values have length {} but "
                       " simulated values have length {}".format(len(obs),
                                                                 len(sim)))
                raise ValueError(msg)
        except ValueError:
            raise

        if self.weights is None:
            self.weights = np.ones(len(obs))

        try:
            if len(self.weights) != len(obs):
                msg = ("Length mismatch! Weights values have length {} but "
                       " observed values have length {}".format(
                            len(self.weights), len(obs))
                       )
                raise ValueError(msg)

        except ValueError:
            raise

        # calculate the likelihood
        residuals = np.asarray(obs) - np.asarray(sim)
        residuals *= self.weights
        var_obs = np.asarray(obs).var()
        var_res = residuals.var()
        likelihood = (1 - (var_res / var_obs))

        if np.isinf(likelihood):
            if np.isneginf(likelihood):
                return -1e10
            else:
                return 1e10

        return likelihood

class GaussianLogLikelihood():
    def __init__(self, var, threshold=0.1, weights=None):
        """
        :param var: the scalar (data-) variance; float
        :param threshold: the threshold to use, given as a fraction of
            simulations that are accepted (the actual likelihood value for
            the threshold is inferred during the tuning phase of MLGLUE);
            float between 0 and 1
        :param weights: a list-like of weights for the individual observations
            (if None, all weights are set to 1.) where the length needs
            to be equal to the number of observations, list-like of floats
            or None
        """

        self.threshold = threshold
        self.var = var
        self.weights = weights

        return

    def likelihood(self, obs=None, sim=None):
        """
        Compute the Gaussian log-likelihood
        
        :param obs: a 1D list-like corresponding to the observation y-values;
            list-like
        :param sim: a 1D list-like corresponding to the simulation y-values;
            list-like
        
        :return likelihood: a float value for the likelihood
        """

        try:
            if obs is None and sim is None:
                msg = ("No observations and no simulated values are given!")
                raise ValueError(msg)
            elif obs is None and sim is not None:
                msg = ("No observations are given!")
                raise ValueError(msg)
            elif obs is not None and sim is None:
                msg = ("No simulated values are given!")
                raise ValueError(msg)
        except ValueError:
            raise

        try:
            if len(sim) != len(obs):
                msg = ("Length mismatch! Observed values have length {} but "
                       " simulated values have length {}".format(len(obs),
                                                                 len(sim)))
                raise ValueError(msg)
        except ValueError:
            raise

        if self.weights is None:
            self.weights = np.ones(len(obs))

        if len(self.weights) != len(obs):
            self.weights = np.ones(len(obs))

        try:
            if len(self.weights) != len(obs):
                msg = ("Length mismatch! Weights values have length {} but"
                       " observed values have length {}".format(
                            len(self.weights), len(obs))
                       )
                raise ValueError(msg)

        except ValueError:
            raise

        # calculate the likelihood
        residuals = np.asarray(obs) - np.asarray(sim)
        residuals *= self.weights

        loglike = (- len(obs)/2) * (np.log(2*np.pi*self.var)) -\
                  (1/(2*self.var)) * np.sum(residuals ** 2)

        if np.isinf(loglike):
            if np.isneginf(loglike):
                return -1e10
            else:
                return 1e10

        return loglike

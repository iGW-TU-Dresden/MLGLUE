import numpy as np

class InverseErrorVarianceLikelihood():
    def __init__(self, threshold=0.1, T=2., weights=None):
        """The Inverse Variance Likelihood class.

        This class represents the Inverse Variance Likelihood function with
        its utilities. It is computed as

        .. math:: L(\\theta | Y) = (\\sigma_e^2)^{-T}

        where :math:`L(\\cdot)` is the likelihood function,
        :math:`\\theta` are model parameters, :math:`Y` are observations,
        :math:`\\sigma_e^2` is the error or residual variance, and
        :math:`T` is the shape parameter.

        Parameters
        ----------
        threshold : float
            The threshold to use, given as a fraction of simulations that
            are accepted (the actual likelihood value for the threshold is
            inferred during the tuning phase of MLGLUE). The value has to
            be in the range (0, 1).
        T : float
            A shape parameter of the likelihood function. When :math:`T=0`,
            every simulation will have equal likelihood. When :math:`T \\to \\infty`,
            the emphasis will be put on the single best simulation. A value
            of :math:`T=1` is often used.
        weights : 1D array-like of float
            The weights of the observations / simulated observation
            equivalents. Note that those weights are not checked further
            and are just used as provided.

        Attributes
        ---------- 
        threshold : float
            The threshold to use, given as a fraction of simulations that
            are accepted (the actual likelihood value for the threshold is
            inferred during the tuning phase of MLGLUE). The value has to
            be in the range (0, 1).
        T : float
            A shape parameter of the likelihood function. When :math:`T=0`,
            every simulation will have equal likelihood. When :math:`T \\to \\infty`,
            the emphasis will be put on the single best simulation. A value
            of :math:`T=1` is often used.
        weights : 1D array-like of float
            The weights of the observations / simulated observation
            equivalents. Note that those weights are not checked further
            and are just used as provided.
        """

        self.threshold = threshold
        self.T = T
        self.weights = weights

        return

    def likelihood(self, obs=None, sim=None):
        """Compute the Inverse Variance Likelihood
        
        Compute the Inverse Variance Likelihood: 

        .. math:: L(\\theta|Y) = (\\sigma_e^2)^{-T}

        where :math:`L(\\cdot)` is the likelihood function,
        :math:`\\theta` are model parameters, :math:`Y` are observations,
        :math:`\\sigma_e^2` is the error or residual variance, and
        :math:`T` is the shape parameter.

        Parameters
        ----------
        obs : 1D array-like of float
            The observations of the system.
        sim : 1D array-like of float
            The simulated observation equivalents, simulated by the model.

        Returns
        -------
        likelihood : float
            The likelihood computed from observations and simulated
            observation equivalents.
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

        # handle infinite likelihood values
        # such values should not completely break the algorithm but either
        # be very large negative or very large positive numbers
        if np.isinf(likelihood):
            if np.isneginf(likelihood):
                return -1e10
            else:
                return 1e10

        return likelihood
    
class InverseErrorVarianceLikelihood_bias():
    def __init__(self, threshold=0.1, T=2., weights=None):
        """The Inverse Variance Likelihood classwith bias term.

        This class represents the Inverse Variance Likelihood function with
        a bias term along its utilities. It is computed as

        .. math::
            L_{\\ell}(\\theta | Y') = \\left( \\frac{\\sum_{j=1}^m (y_j - y'_j + \\mu_{B,\\ell,j} )^2}{m - 2} \\right)^{-T}

        where :math:`L(\\cdot)` is the likelihood function,
        :math:`\\theta` are model parameters, :math:`Y` are observations,
        :math:`\\mu_{B,\\ell}` is the bias up to level :math:`\\ell`,
        :math:`m` is the number of observations, and :math:`T` is the shape
        parameter.

        Parameters
        ----------
        threshold : float
            The threshold to use, given as a fraction of simulations that
            are accepted (the actual likelihood value for the threshold is
            inferred during the tuning phase of MLGLUE). The value has to
            be in the range (0, 1).
        T : float
            A shape parameter of the likelihood function. When :math:`T=0`,
            every simulation will have equal likelihood. When :math:`T \\to \\infty`,
            the emphasis will be put on the single best simulation. A value
            of :math:`T=1` is often used.
        weights : 1D array-like of float
            The weights of the observations / simulated observation
            equivalents. Note that those weights are not checked further
            and are just used as provided.

        Attributes
        ---------- 
        threshold : float
            The threshold to use, given as a fraction of simulations that
            are accepted (the actual likelihood value for the threshold is
            inferred during the tuning phase of MLGLUE). The value has to
            be in the range (0, 1).
        T : float
            A shape parameter of the likelihood function. When :math:`T=0`,
            every simulation will have equal likelihood. When :math:`T \\to \\infty`,
            the emphasis will be put on the single best simulation. A value
            of :math:`T=1` is often used.
        weights : 1D array-like of float
            The weights of the observations / simulated observation
            equivalents. Note that those weights are not checked further
            and are just used as provided.
        """

        self.threshold = threshold
        self.T = T
        self.weights = weights

        return

    def likelihood(self, obs=None, sim=None, bias=None):
        """Compute the Inverse Variance Likelihood
        
        Compute the Inverse Variance Likelihood: 

        .. math::
            L_{\\ell}(\\theta | Y') = \\left( \\frac{\\sum_{j=1}^m (y_j - y'_j + \\mu_{B,\\ell,j} )^2}{m - 2} \\right)^{-T}

        where :math:`L(\\cdot)` is the likelihood function,
        :math:`\\theta` are model parameters, :math:`Y` are observations,
        :math:`\\mu_{B,\\ell}` is the bias up to level :math:`\\ell`,
        :math:`m` is the number of observations, and :math:`T` is the shape
        parameter.

        Parameters
        ----------
        obs : 1D array-like of float
            The observations of the system.
        sim : 1D array-like of float
            The simulated observation equivalents, simulated by the model.
        bias : 1D array-like of float
            The bias on the level to which the current call to the
            likelihood belongs. Has the same length as obs and sim.

        Returns
        -------
        likelihood : float
            The likelihood computed from observations, simulated
            observation equivalents, and the bias term.
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

        try:
            # if no bias is given, we assume it is 0
            if bias is None:
                bias = np.zeros_like(sim)
            if len(sim) != len(bias):
                msg = ("Length mismatch! Bias vector does not match!")
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
        residuals = np.asarray(obs) - np.asarray(sim) + np.asarray(bias)
        residuals *= weights
        ssr = np.sum(residuals ** 2)
        likelihood = (ssr / (len(obs) - 2)) ** (-self.T)

        # handle infinite likelihood values
        # such values should not completely break the algorithm but either
        # be very large negative or very large positive numbers
        if np.isinf(likelihood):
            if np.isneginf(likelihood):
                return -1e10
            else:
                return 1e10

        return likelihood

class RelativeVarianceLikelihood():
    def __init__(self, threshold=0.1, weights=None):
        """The Relative Variance Likelihood class.

        This class represents the Relative Variance Likelihood function
        with its utilities. It is computed as

        .. math:: L(\\theta | Y) = 1 - \\frac{\\sigma_e^2}{\\sigma_{obs}^2}

        where :math:`L(\\cdot)` is the likelihood function,
        :math:`\\theta` are model parameters, :math:`Y` are observations,
        :math:`\\sigma_e^2` is the variance of errors or residuals, and
        :math:`\\sigma_{obs}^2` is the variance of observed values.

        Parameters
        ----------
        threshold : float
            The threshold to use, given as a fraction of simulations that
            are accepted (the actual likelihood value for the threshold is
            inferred during the tuning phase of MLGLUE). The value has to
            be in the range (0, 1).
        weights : 1D array-like of float
            The weights of the observations / simulated observation
            equivalents. Note that those weights are not checked further
            and are just used as provided.

        Attributes
        ---------- 
        threshold : float
            The threshold to use, given as a fraction of simulations that
            are accepted (the actual likelihood value for the threshold is
            inferred during the tuning phase of MLGLUE). The value has to
            be in the range (0, 1).
        weights : 1D array-like of float
            The weights of the observations / simulated observation
            equivalents. Note that those weights are not checked further
            and are just used as provided.
        """

        self.threshold = threshold
        self.weights = weights

        return

    def likelihood(self, obs=None, sim=None):
        """Compute the Relative Variance Likelihood
        
        Compute the Relative Variance Likelihood:

        .. math:: L(\\theta | Y) = 1 - \\frac{\\sigma_e^2}{\\sigma_{obs}^2}

        where :math:`L(\\cdot)` is the likelihood function,
        :math:`\\theta` are model parameters, :math:`Y` are observations,
        :math:`\\sigma_e^2` is the variance of errors or residuals, and
        :math:`\\sigma_{obs}^2` is the variance of observed values.

        Parameters
        ----------
        obs : 1D array-like of float
            The observations of the system.
        sim : 1D array-like of float
            The simulated observation equivalents, simulated by the model.

        Returns
        -------
        likelihood : float
            The likelihood computed from observations and simulated
            observation equivalents.
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

        # handle infinite likelihood values
        # such values should not completely break the algorithm but either
        # be very large negative or very large positive numbers
        if np.isinf(likelihood):
            if np.isneginf(likelihood):
                return -1e10
            else:
                return 1e10

        return likelihood

class GaussianLogLikelihood():
    def __init__(self, var, threshold=0.1, weights=None):
        """The Gaussian log-likelihood class.

        This class represents the Gaussian log-likelihood function
        with its utilities. It is computed as

        .. math:: L(\\theta | Y) = - \\frac{n}{2}\\ln(2\\pi) - \\frac{n}{2}\\ln(\\sigma^2) - \\frac{1}{2}\\sigma^{-2} \\times \\sum_{i=1}^n (y'_i(\\theta) - y_i)^2

        where :math:`L(\\cdot)` is the log-likelihood function,
        :math:`\\theta` are model parameters, :math:`y_i` are observations,
        :math:`\\sigma^2` is the (theoretical) variance of errors or residuals,
        :math:`y'_i(\\cdot)` is the :math:`i`-th model output corresponding
        to the :math:`i`-th observation.

        Parameters
        ----------
        threshold : float
            The threshold to use, given as a fraction of simulations that
            are accepted (the actual likelihood value for the threshold is
            inferred during the tuning phase of MLGLUE). The value has to
            be in the range (0, 1).
        var : float
            The (theoretical) error variance of the likelihood function.
        weights : 1D array-like of float
            The weights of the observations / simulated observation
            equivalents. Note that those weights are not checked further
            and are just used as provided.

        Attributes
        ---------- 
        threshold : float
            The threshold to use, given as a fraction of simulations that
            are accepted (the actual likelihood value for the threshold is
            inferred during the tuning phase of MLGLUE). The value has to
            be in the range (0, 1).
        var : float
            The (theoretical) error variance of the likelihood function.
        weights : 1D array-like of float
            The weights of the observations / simulated observation
            equivalents. Note that those weights are not checked further
            and are just used as provided.
        """

        self.threshold = threshold
        self.var = var
        self.weights = weights

        return

    def likelihood(self, obs=None, sim=None):
        """Compute the Gaussian log-likelihood
        
        Compute the Gaussian log-likelihood: 

        .. math:: L(\\theta | Y) = - \\frac{n}{2}\\ln(2\\pi) - \\frac{n}{2}\\ln(\\sigma^2) - \\frac{1}{2}\\sigma^{-2} \\times \\sum_{i=1}^n (y'_i(\\theta) - y_i)^2

        where :math:`L(\\cdot)` is the log-likelihood function,
        :math:`\\theta` are model parameters, :math:`y_i` are observations,
        :math:`\\sigma^2` is the (theoretical) variance of errors or residuals,
        :math:`y'_i(\\cdot)` is the :math:`i`-th model output corresponding
        to the :math:`i`-th observation.

        Parameters
        ----------
        obs : 1D array-like of float
            The observations of the system.
        sim : 1D array-like of float
            The simulated observation equivalents, simulated by the model.

        Returns
        -------
        likelihood : float
            The likelihood computed from observations and simulated
            observation equivalents.
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

        # handle infinite likelihood values
        # such values should not completely break the algorithm but either
        # be very large negative or very large positive numbers
        if np.isinf(loglike):
            if np.isneginf(loglike):
                return -1e10
            else:
                return 1e10

        return loglike

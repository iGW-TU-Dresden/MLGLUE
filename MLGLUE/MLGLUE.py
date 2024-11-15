import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import ray
from ray.util.multiprocessing import Pool

class MLGLUE():
    """The MLGLUE class.
    
    This is the basic class of the MLGLUE implementation. It is used to
    define all general settings of MLGLUE sampling for a given case
    such as the model function, the likelihood function, parameter
    samples (or settings for internal sample generation if no samples
    are given by the user directly), etc.

    Parameters
    ----------
    model : Callable
        The callable representing the model to which MLGLUE should be
        applied. See the Notes section below for further details.
    likelihood
        An instance of an MLGLUE likelihood object. See the Notes
        section below for further details.
    lower_bounds : 1D array-like of float
        The lower bounds of the uniform distribution over the model
        parameters. This attribute is ignored if `samples` are directly
        supplied. `lower_bounds` has to have shape (n_parameters,).
    upper_bounds : 1D array-like of float
        The upper bounds of the uniform distribution over the model
        parameters. This attribute is ignored if `samples` are directly
        supplied. `upper_bounds` has to have shape (n_parameters,).
    n_samples : int
        The total number of parameter samples to draw from the uniform
        prior distribution. Note that this includes the samples used
        for tuning! This attribute is ignored if `samples` are directly
        supplied.
    samples : (tuple of) 2D array-like of float, optional
        The prior parameter samples, which can optionally be supplied.
        If `samples` are given, `lower_bounds`, `upper_bounds`, and
        `n_samples` are ignored. Default is None. If a 2D array-like,
        it is considered the full set of parameter samples, including
        the samples used for tuning. If a tuple of 2D array-like, the
        first element of the tuple is considered as tuning samples and
        the second element as the regular samples. Every 2D array-like
        has to have shape (n_samples, n_parameters).
    tuning : float
        The tuning fraction (0. < `tuning` < 1.). The rounded result
        (int(n_samples * tuning) is used to split the samples into the
        two parts.
    n_levels : int
        The number of levels in the hierarchy.
    coarsening_factor : float
        For the case of a geometric series of resolutions in the
        hierarchy, `coarsening_factor` represents the coarsening of the
        resolution when going from level (l) to level (l-1).
    obs : 1D array-like of float
        The observations for which the model simulates values.
    thresholds : 1D array-like of float, optional
        The level-dependent likelihood thresholds to use. Has to have
        shape (n_levels,). If `thresholds` is given, the tuning phase
        is skipped. Note that this has an effect on the definition of
        the parameter samples! Default is None.
    multiprocessing : bool
        Whether to use multiprocessing using the Ray package or not.
    n_processors : int, optional
        The number of CPUs to use if `multiprocessing` is enabled. By
        default, all processors are used (`n_processors`=None).
    savefigs
        Whether to save variance analysis figures or not. If None,
        figures will not be saved. If str, figures will be saved as png
        with the str as identifier.
    hiearchy_analysis : bool
        Whether hiararchy analysis is strict or not. If not strict 
        (False), results of variance and mean analysis are printed to
        the screen but MLGLUE continues independently of the result. If
        strict (True), results are also printed to the screen but
        MLGLUE is stopped if the variances and mean values between
        levels (l-1, l) is larger than on level (l) and / or if the
        variances or mean values between levels do not decay
        monotonically.
    
    Attributes
    ----------
    model : Callable
        The callable representing the model to which MLGLUE should be
        applied. See the Notes section below for further details.
    likelihood
        An instance of an MLGLUE likelihood object. See the Notes
        section below for further details.
    lower_bounds : 1D array-like of float
        The lower bounds of the uniform distribution over the model
        parameters. This attribute is ignored if `samples` are directly
        supplied. `lower_bounds` has to have shape (n_parameters,).
    upper_bounds : 1D array-like of float
        The upper bounds of the uniform distribution over the model
        parameters. This attribute is ignored if `samples` are directly
        supplied. `upper_bounds` has to have shape (n_parameters,).
    n_samples : int
        The total number of parameter samples to draw from the uniform
        prior distribution. Note that this includes the samples used
        for tuning! This attribute is ignored if `samples` are directly
        supplied.
    samples : 2D array-like of float
        The prior parameter samples, which can optionally be supplied.
        Note that this includes the samples used for tuning! If
        `samples` are given, `lower_bounds`, `upper_bounds`, and
        `n_samples` are ignored. Has to have shape
        (n_samples, n_parameters).
    samples_tuning : 2D array-like of float
        The (prior) parameter samples used for tuning. If
        `samples_tuning` are given, `lower_bounds`, `upper_bounds`, and
        `n_samples` are ignored. Has to have shape (n_samples,
        n_parameters).
    samples_tuning : 2D array-like of float
        The (prior) parameter samples used for sampling. If
        `samples_sampling` are given, `lower_bounds`, `upper_bounds`,
        and `n_samples` are ignored. Has to have shape (n_samples,
        n_parameters).
    tuning : float
        The tuning fraction (0. < `tuning` < 1.). The rounded result
        (int(n_samples * tuning) is used to split the samples into the
        two parts.
    n_levels : int
        The number of levels in the hierarchy.
    coarsening_factor : float
        For the case of a geometric series of resolutions in the
        hierarchy, `coarsening_factor` represents the coarsening of the
        resolution when going from level (l) to level (l-1).
    obs : 1D array-like of float
        The observations for which the model simulates values.
    thresholds : 1D array-like of float
        The level-dependent likelihood thresholds to use. Has to have
        shape (n_levels,). If `thresholds` is given, the tuning phase
        is skipped. Note that this has an effect on the definition of
        the parameter samples! Default is None.
    multiprocessing : bool
        Whether to use multiprocessing using the Ray package or not.
    n_processors : int
        The number of CPUs to use if `multiprocessing` is enabled. By
        default, all processors are used (`n_processors`=None).
    savefigs
        Whether to save variance analysis figures or not. If None,
        figures will not be saved. If str, figures will be saved as png
        with the str as identifier.
    hiearchy_analysis : bool
        Whether hiararchy analysis is strict or not. If not strict 
        (False), results of variance and mean analysis are printed to
        the screen but MLGLUE continues independently of the result. If
        strict (True), results are also printed to the screen but
        MLGLUE is stopped if the variances and mean values between
        levels (l-1, l) is larger than on level (l) and / or if the
        variances or mean values between levels do not decay
        monotonically.
    selected_samples : 2D array-like of float
        The array of selected samples that are accepted on the highest
        level; has individual samples as rows and variables / model
        parameters as columns.
    results : 2D array-like of float
        Holds simulated observation equivalents corresponding to all
        posterior samples; has shape (len(selected_samples), len(obs)).
    results_analysis : 3D array-like of float
        Holds simulated observation equivalents corresponding to all
        posterior samples on all levels; has shape (n_levels,
        len(selected_samples), len(obs)).
    results_analysis_tuning : 3D array-like of float
        Holds simulated observation equivalents corresponding to all
        tuning samples (except for tuning samples that result in an
        error or NaN returned by the model callable) on all levels;
        has the tuning samples in the first dimension, the levels in
        the second dimension, and the simulated values in the third
        dimension.
    likelihoods : 1D array-like of float
        The likelihood values correpsonding to the selected samples.
    normalized_likelihoods : 1D array-like of float
        Normalized likelihood values used for the computation of
        uncertainty estimates.
    likelihoods_tuning : 2D array-like of float
        The likelihood values on all levels for all tuning samples
        (except for tuning samples that result in an error or NaN
        returned by the model callable) on all levels; has the levels
        in the rows and tuning samples in columns.
    highest_level_calls : 1D array-like of int
        A list with the number of ones equal to the number of calls
        made to the model on the highest level. This is implemented
        like that currently as a list can be shared across processes / 
        workers. A single variable (e.g., an int) could not be shared
        this way. This will be improved in the future.        
    
    Notes
    -----
    The callable for the `model` attribute has to accept the following
    arguments: parameters (1D list-like of floats representing model
    parameters), level (0-based integer representing the level index),
    n_levels (integer representing the total number of levels), obs (1D
    list-like of floats representing observations), likelihood (the
    MLGLUE likelihood function). The callable has to return a float
    corresponding to the likelihood of the given parameter sample and
    return simulation results (these results can either be only
    simulated observation equivalents or other simulation results; if
    other simulation results are given as well, the corresponding
    weight in the likelihood function should be set to zero). The
    likelihood value should be computed using a likelihood function
    implemented in this package; user-defined likelihood functions can
    be used as well but are not tested. The simulated values have to be
    of a type that can be appended to a list but do not have to have a
    certain structure otherwise.
    The object instance for the `likelihood` attribute should then have
    a `likelihood` method (see the Examples section for further
    details). The likelihood method has to accept the following
    arguments: obs (1D list-like of floats representing observations),
    sim (1D list-like of floats representing simulated observation
    equivalents). Using a likelihood included in the present package
    already ensures this structure.
    If the model function only has one level, it should be the finest /
    target level.
    """
    def __init__(
            self,
            model,
            likelihood=None,
            lower_bounds=None,
            upper_bounds=None,
            n_samples=1000,
            samples=None,
            tuning=0.2,
            n_levels=1,
            coarsening_factor=2,
            obs=None,
            thresholds=None,
            multiprocessing=False,
            n_processors=None,
            hierarchy_analysis=True,
            savefigs="my_model",
            include_bias=False
    ):
        """The MLGLUE class.

        This is the basic class of the MLGLUE implementation. It is used to
        define all general settings of MLGLUE sampling for a given case
        such as the model function, the likelihood function, parameter
        samples (or settings for internal sample generation if no samples
        are given by the user directly), etc.

        Parameters
        ----------
        model : Callable
            The callable representing the model to which MLGLUE should be
            applied. See the Notes section below for further details.
        likelihood
            An instance of an MLGLUE likelihood object. See the Notes
            section below for further details.
        lower_bounds : 1D array-like of float
            The lower bounds of the uniform distribution over the model
            parameters. This attribute is ignored if `samples` are directly
            supplied. `lower_bounds` has to have shape (n_parameters,).
        upper_bounds : 1D array-like of float
            The upper bounds of the uniform distribution over the model
            parameters. This attribute is ignored if `samples` are directly
            supplied. `upper_bounds` has to have shape (n_parameters,).
        n_samples : int
            The total number of parameter samples to draw from the uniform
            prior distribution. Note that this includes the samples used
            for tuning! This attribute is ignored if `samples` are directly
            supplied.
        samples : (tuple of) 2D array-like of float, optional
            The prior parameter samples, which can optionally be supplied.
            If `samples` are given, `lower_bounds`, `upper_bounds`, and
            `n_samples` are ignored. Default is None. If a 2D array-like,
            it is considered the full set of parameter samples, including
            the samples used for tuning. If a tuple of 2D array-like, the
            first element of the tuple is considered as tuning samples and
            the second element as the regular samples. Every 2D array-like
            has to have shape (n_samples, n_parameters).
        tuning : float
            The tuning fraction (0. < `tuning` < 1.). The rounded result
            (int(n_samples * tuning) is used to split the samples into the
            two parts.
        n_levels : int
            The number of levels in the hierarchy.
        coarsening_factor : float
            For the case of a geometric series of resolutions in the
            hierarchy, `coarsening_factor` represents the coarsening of the
            resolution when going from level (l) to level (l-1).
        obs : 1D array-like of float
            The observations for which the model simulates values.
        thresholds : 1D array-like of float, optional
            The level-dependent likelihood thresholds to use. Has to have
            shape (n_levels,). If `thresholds` is given, the tuning phase
            is skipped. Note that this has an effect on the definition of
            the parameter samples! Default is None.
        multiprocessing : bool
            Whether to use multiprocessing using the Ray package or not.
        n_processors : int, optional
            The number of CPUs to use if `multiprocessing` is enabled. By
            default, all processors are used (`n_processors`=None).
        savefigs
            Whether to save variance analysis figures or not. If None,
            figures will not be saved. If str, figures will be saved as png
            with the str as identifier.
        hiearchy_analysis : bool
            Whether hiararchy analysis is strict or not. If not strict 
            (False), results of variance and mean analysis are printed to
            the screen but MLGLUE continues independently of the result. If
            strict (True), results are also printed to the screen but
            MLGLUE is stopped if the variances and mean values between
            levels (l-1, l) is larger than on level (l) and / or if the
            variances or mean values between levels do not decay
            monotonically.
        include_bias : bool
            Whether to include the computation of bias vectors or not. If
            included (True), a likelihood must be used which accepts a
            bias vector (e.g., InverseErrorVarianceLikelihood_bias). Bias
            is computed for lower-level models w.r.t. the highest-level
            model.

        Attributes
        ----------
        model : Callable
            The callable representing the model to which MLGLUE should be
            applied. See the Notes section below for further details.
        likelihood
            An instance of an MLGLUE likelihood object. See the Notes
            section below for further details.
        lower_bounds : 1D array-like of float
            The lower bounds of the uniform distribution over the model
            parameters. This attribute is ignored if `samples` are directly
            supplied. `lower_bounds` has to have shape (n_parameters,).
        upper_bounds : 1D array-like of float
            The upper bounds of the uniform distribution over the model
            parameters. This attribute is ignored if `samples` are directly
            supplied. `upper_bounds` has to have shape (n_parameters,).
        n_samples : int
            The total number of parameter samples to draw from the uniform
            prior distribution. Note that this includes the samples used
            for tuning! This attribute is ignored if `samples` are directly
            supplied.
        samples : 2D array-like of float
            The prior parameter samples, which can optionally be supplied.
            Note that this includes the samples used for tuning! If
            `samples` are given, `lower_bounds`, `upper_bounds`, and
            `n_samples` are ignored. Has to have shape
            (n_samples, n_parameters).
        samples_tuning : 2D array-like of float
            The (prior) parameter samples used for tuning. If
            `samples_tuning` are given, `lower_bounds`, `upper_bounds`, and
            `n_samples` are ignored. Has to have shape (n_samples,
            n_parameters).
        samples_tuning : 2D array-like of float
            The (prior) parameter samples used for sampling. If
            `samples_sampling` are given, `lower_bounds`, `upper_bounds`,
            and `n_samples` are ignored. Has to have shape (n_samples,
            n_parameters).
        tuning : float
            The tuning fraction (0. < `tuning` < 1.). The rounded result
            (int(n_samples * tuning) is used to split the samples into the
            two parts.
        n_levels : int
            The number of levels in the hierarchy.
        coarsening_factor : float
            For the case of a geometric series of resolutions in the
            hierarchy, `coarsening_factor` represents the coarsening of the
            resolution when going from level (l) to level (l-1).
        obs : 1D array-like of float
            The observations for which the model simulates values.
        thresholds : 1D array-like of float
            The level-dependent likelihood thresholds to use. Has to have
            shape (n_levels,). If `thresholds` is given, the tuning phase
            is skipped. Note that this has an effect on the definition of
            the parameter samples! Default is None.
        multiprocessing : bool
            Whether to use multiprocessing using the Ray package or not.
        n_processors : int
            The number of CPUs to use if `multiprocessing` is enabled. By
            default, all processors are used (`n_processors`=None).
        savefigs
            Whether to save variance analysis figures or not. If None,
            figures will not be saved. If str, figures will be saved as png
            with the str as identifier.
        hiearchy_analysis : bool
            Whether hiararchy analysis is strict or not. If not strict 
            (False), results of variance and mean analysis are printed to
            the screen but MLGLUE continues independently of the result. If
            strict (True), results are also printed to the screen but
            MLGLUE is stopped if the variances and mean values between
            levels (l-1, l) is larger than on level (l) and / or if the
            variances or mean values between levels do not decay
            monotonically.
        selected_samples : 2D array-like of float
            The array of selected samples that are accepted on the highest
            level; has shape (n_selected_samples, n_model_parameters)
        results : 2D array-like of float
            Holds simulated observation equivalents corresponding to all
            posterior samples; has shape (len(selected_samples), len(obs)).
        results_analysis : 3D array-like of float
            Holds simulated observation equivalents corresponding to all
            posterior samples on all levels; has shape (n_levels,
            len(selected_samples), len(obs)).
        results_analysis_tuning : 3D array-like of float
            Holds simulated observation equivalents corresponding to all
            tuning samples (except for tuning samples that result in an
            error or NaN returned by the model callable) on all levels;
            has shape (n_levels, len(selected_samples_tuning), len(obs))
        likelihoods : 1D array-like of float
            The likelihood values correpsonding to the selected samples.
        normalized_likelihoods : 1D array-like of float
            Normalized likelihood values used for the computation of
            uncertainty estimates.
        likelihoods_tuning : 2D array-like of float
            The likelihood values on all levels for all tuning samples
            (except for tuning samples that result in an error or NaN
            returned by the model callable) on all levels; has shape
            (n_levels, len(selected_samples_tuning))
        highest_level_calls : 1D array-like of int
            A list with the number of ones equal to the number of calls
            made to the model on the highest level. This is implemented
            like that currently as a list can be shared across processes / 
            workers. A single variable (e.g., an int) could not be shared
            this way. This will be improved in the future.
        include_bias : bool
            Whether to include the computation of bias vectors or not. If
            included (True), a likelihood must be used which accepts a
            bias vector (e.g., InverseErrorVarianceLikelihood_bias). Bias
            is computed for lower-level models w.r.t. the highest-level
            model.
        bias : 2D array-like of float
            Holds the bias vectors for all levels. Has shape (n_levels,
            len(obs)).
        
        Notes
        -----
        The callable for the `model` attribute has to accept the following
        arguments: parameters (1D list-like of floats representing model
        parameters), level (0-based integer representing the level index),
        n_levels (integer representing the total number of levels), obs (1D
        list-like of floats representing observations), likelihood (the
        MLGLUE likelihood function). The callable has to return a float
        corresponding to the likelihood of the given parameter sample and
        return simulation results (these results can either be only
        simulated observation equivalents or other simulation results; if
        other simulation results are given as well, the corresponding
        weight in the likelihood function should be set to zero). The
        likelihood value should be computed using a likelihood function
        implemented in this package; user-defined likelihood functions can
        be used as well but are not tested. The simulated values have to be
        of a type that can be appended to a list but do not have to have a
        certain structure otherwise.
        The object instance for the `likelihood` attribute should then have
        a `likelihood` method (see the Examples section for further
        details). The likelihood method has to accept the following
        arguments: obs (1D list-like of floats representing observations),
        sim (1D list-like of floats representing simulated observation
        equivalents). Using a likelihood included in the present package
        already ensures this structure.
        If the model function only has one level, it should be the finest /
        target level.

        """
        self.model = model
        self.likelihood = likelihood
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.n_samples = n_samples
        self.samples = samples
        self.samples_tuning = None
        self.samples_sampling = None
        self.tuning = tuning
        self.thresholds = None
        self.n_levels = n_levels
        self.coarsening_factor = coarsening_factor
        self.obs = obs
        self.multiprocessing = multiprocessing
        self.n_processors = n_processors
        self.hierarchy_analysis = hierarchy_analysis
        self.savefigs = savefigs
        self.include_bias = include_bias

        # initialize output data structures
        self.normalized_likelihoods = None
        self.results = [] # --> this holds only the final results (i.e., from the finest level)
        self.selected_samples = []
        self.likelihoods = []
        self.likelihoods_tuning = [[] for i in range(self.n_levels)]
        self.results_analysis = [[] for i in range(self.n_levels)] # --> holds results from all levels during sampling
        self.results_analysis_tuning = [[] for i in range(self.n_levels)] # --> holds results from all levels during tuning
        self.highest_level_calls = [] # --> holds identifiers for highest level calls (1 corresponds to a highest level call)
        self.bias = np.zeros((self.n_levels, len(self.obs))) # --> holds bias vectors

        if thresholds is not None:
            self.thresholds = thresholds
            self.thresholds_predefined = True

            # print warning to screen if bias should be included
            if self.include_bias:
                print("- - - -\n")
                print("- -\n")
                print("If bias should be included, thresholds cannot be \
                      pre-defined.\n")
                print("- -\n")
                print("- - - -\n")
        else:
            self.thresholds = []
            self.thresholds_predefined = False

        return
    
    def MLGLUE_tuning(self, samples):
        """Single-core MLGLUE tuning phase.

        Perform the tuning phase of MLGLUE without using multiprocessing /
        Ray.
        
        Parameters
        ----------
        samples : 2D list-like of float
            The parameter samples with which to perform tuning. Has to have
            the individual samples as rows and variables / model parameters
            as columns.

        Returns
        -------
        None

        Notes
        -----
        This is implemented as an instance method; it cannot be used
        independently of an MLGLUE instance. Therefore, results are not
        returned by this function but are rather stored in the class /
        instance attributes.
        """

        # set the internal level index to 0 (i.e., start sample evaluation
        # on level 0)
        run_id = 0

        # self.likelihoods_tuning[level_].append(likelihood_)
        # self.results_analysis_tuning[level_].append(results)

        # iterate over all samples
        for sample in samples:
            sample_result = self.evaluate_sample_tuning(
                sample=sample,
                run_id=run_id
            )

            # if there was an error with on any level, sample_result is
            # None. in that case, continue to the next sample
            if sample_result is None:
                # increment the run_id
                run_id += 1
                continue
            
            # otherwise, handle the results returned above
            else:
                # the order of elements in the sample_result tuple is:
                # likelihoods_sample, results_analysis_tuning_sample

                # append level-dependent results to internal data
                # structures
                for level_ in range(self.n_levels):
                    # append likelihoods
                    self.likelihoods_tuning[level_].append(
                        sample_result[0][level_]
                    )
                    # append results
                    self.results_analysis_tuning[level_].append(
                        sample_result[1][level_]
                    )

                # increment the run_id
                run_id += 1

        return

    def analyze_variances_likelihoods(self, raise_error=True):
        """Analyze likelihood variances.

        Analyze the relationships between levels in terms of variances:
        (1) variances of likelihood values on each individual level (2)
        covariances of likelihoods across levels (3) variances of the
        difference between likelihood values on subsequent levels.

        Specifically, the following expression is evaluated for all levels:
        
        .. math:: \mathbb{V}[L_{\\ell} - L_{\\ell-1}] = \mathbb{V}[L_{\\ell}] + \mathbb{V}[L_{\\ell-1}] - 2 \cdot Cov[L_{\\ell}-L_{\\ell-1}]

        where :math:`L_{\\ell}` is the random variable representing
        likelihood values on level :math:`\\ell` form the tuning phase.
        Then, for :math:`\mathbb{V}[L_{\\ell}-L_{\\ell-1}]` to decay
        monotonically, :math:`2 \cdot Cov[L_{\\ell}-L_{\\ell-1}]` has to be
        larger than :math:`\mathbb{V}[L_{\\ell-1}]`, which implies that two
        subsequent levels need to be sufficiently correlated.

        Parameters
        ----------
        raise_error : bool
            Whether variance analysis is strict or not. If not strict 
            (False), results of variance analysis are printed to the screen
            but MLGLUE continues independently of the result. If strict
            (True), results are also printed to the screen but MLGLUE is
            stopped if the variances between levels (l-1, l) is larger than
            on level (l) and / or if the variances between levels do not
            decay monotonically.

        Returns
        -------
        None
        """

        # get number of likelihoods in every level
        # this number is identical on every level
        n_vals = len(self.likelihoods_tuning[0, :])

        # compute the variance of the likelihood values on each level
        vars_within_levels = []
        for i in self.likelihoods_tuning:
            # calculate the variance within the level
            var_ = np.var(i, axis=0, ddof=1)
            # append variance to list
            vars_within_levels.append(var_)

        # calculate covariance between / across subsequent levels
        covs_cross_levels = []
        cov = np.cov(
            self.likelihoods_tuning,
            rowvar=True,
            ddof=1
        )
        for i in range(self.n_levels - 1):
            covs_cross_levels.append(cov[i, i+1])

        # calculate correlation between / across subsequent levels
        corrs_cross_levels = []
        corr = np.corrcoef(
            self.likelihoods_tuning,
            rowvar=True,
            ddof=1
        )
        for i in range(self.n_levels - 1):
            corrs_cross_levels.append(corr[i, i+1])

        # calculate right hand side of expression
        rhs = []
        for i in range(self.n_levels - 1):
            # calculate rhs
            rhs_ = vars_within_levels[i] + vars_within_levels[i + 1] - \
                2. * covs_cross_levels[i]
            # append rhs to list
            rhs.append(rhs_)

        # print correlation between subsequent levels
        print(
            "\n\nResults of variance analysis: " + 
            "\nCorrelation between subsequent levels (from lowest to" + 
            " highest level):"
        )
        for i in range(len(corrs_cross_levels)):
            print("   {:1.5f}   (level {}, level {})".format(
                corrs_cross_levels[i],
                i,
                i + 1
                )
            )
        print("Note: those values should INCREASE with increasing level " +
              "indices!")
        print("\n")

        # print variances on different levels
        print( 
            "Variances of likelihoods on all levels (from lowest to" + 
            " highest level):"
        )
        for i in range(len(vars_within_levels)):
            print("   {:1.5f}   (level {})".format(
                vars_within_levels[i],
                i
                )
            )
        print("Note: those values should be approximately constant " + 
              "across all levels!")
        print("\n")

        # initialize list to store booleans representing whether the
        # variance inequality holds or not
        ineq = []
        # initialize list to store booleans representing whether the
        # cross-level variance decays
        decay = []

        # iterate over levels to check variance inequality
        for i in range(self.n_levels - 1):
            if vars_within_levels[i + 1] >= rhs[i]:
                print("The var. inequality holds between levels ",
                      "{} and {}: {:1.5f} >= {:1.5e}".format(
                          i,
                          i + 1,
                          vars_within_levels[i + 1],
                          rhs[i]
                        )
                    )
                ineq.append(True)
            else:
                print("The var. inequality DOES NOT hold between levels ",
                      "{} and {}: {:1.5f} >//= {:1.5e}".format(
                          i,
                          i + 1,
                          vars_within_levels[i + 1],
                          rhs[i]
                        )
                    )
                ineq.append(False)

        # iterate over levels to check cross-level variance decay
        for i in range(self.n_levels - 2):
            if rhs[i] > rhs[i+1]:
                decay.append(True)
            else:
                decay.append(False)

        try:
            if not np.array(ineq).all():
                msg = ("\nThe variance inequality does not hold for all"
                       "two subsequent levels!")
                print(msg)
            else:
                print("\nThe variance inequality holds between all two "
                      "subsequent levels!")
        
            if not np.array(decay).all():
                msg = ("The cross-level variance does not decay "
                       "monotonically!")
                print(msg)
            else:
                print("The cross-level variance decays monotonically!")
        
        except ValueError:
            if raise_error:
                raise
            else:
                pass

        return

    def analyze_means_likelihoods(self, raise_error=True):
        """Analyze likelihood mean values.

        Analyze the relationships between levels in terms of mean values:
        (1) mean values of likelihood values on each individual level (2)
        mean values of the difference between likelihood values on
        subsequent levels.

        Specifically, the following expression is evaluated for all levels:

        .. math:: \mathbb{E}[L_{\\ell} - L_{\\ell-1}] = \mathbb{E}[L_{\\ell}] - \mathbb{E}[L_{\\ell-1}]

        where :math:`L_{\\ell}` is the random variable representing
        likelihood values on level :math:`\\ell` form the tuning phase.
        Then, for :math:`E[L_{\\ell}-L_{\\ell-1}]` to decay
        monotonically, the difference in the mean values of the likelihoods
        on different levels has to decay.

        Parameters
        ----------
        raise_error : bool
            Whether mean value analysis is strict or not. If not strict 
            (False), results of mean value analysis are printed to the
            screen but MLGLUE continues independently of the result. If
            strict (True), results are also printed to the screen but
            MLGLUE is stopped if the mean values of the difference between
            likelihood values on levels (l-1, l) are larger than on level
            (l) and / or if the mean values of the difference between
            likelihood values do not decay monotonically.

        Returns
        -------
        None
        """

        # get number of likelihoods in every level
        # this number is identical on every level
        n_vals = len(self.likelihoods_tuning[0, :])

        # compute mean value of the likelihoods on each level
        means_within_levels = []
        for i in self.likelihoods_tuning:
            # calculate the variance within the level
            mean_ = abs(np.mean(i, axis=0))
            # append variance to list
            means_within_levels.append(mean_)

        # calculate mean of difference between likelihood values on
        # subsequent levels
        rhs = []
        for i in range(self.n_levels - 1):
            rhs_ = abs(means_within_levels[i] - means_within_levels[i + 1])
            rhs.append(rhs_)

        # print mean values of differences of likelihoods between
        # subsequent levels
        print(
            "\n\nResults of mean value analysis: " + 
            "\nMean values of the difference between likelihoods on "
            "subsequent levels (from lowest to highest level):"
        )
        for i in range(len(rhs)):
            print("   {:1.5f}   (level {}, level {})".format(
                rhs[i],
                i,
                i + 1
                )
            )
        print("Note: those values should DECREASE with increasing level " +
              "indices!")
        print("\n")

        # print mean values of likelihoods on all levels
        print(
            "Mean values of the likelihoods on all levels (from lowest " +
            "highest level):"
        )
        for i in range(len(means_within_levels)):
            print("   {:1.5f}   (level {})".format(
                means_within_levels[i],
                i
                )
            )
        print("Note: those values should be approximately constant " + 
              "across all levels!")
        print("\n")

        ineq = []
        decay = []

        # iterate over levels to check conditions
        for i in range(self.n_levels - 1):
            if means_within_levels[i + 1] >= rhs[i]:
                print("The mean value ineq. holds between levels ",
                      "{} and {}: {:1.5f} >= {:1.5e}".format(
                          i,
                          i + 1,
                          means_within_levels[i + 1],
                          rhs[i]
                        )
                    )
                ineq.append(True)
            else:
                print("The mean value ineq. DOES NOT hold between levels ",
                      "{} and {}: {:1.5f} >//= {:1.5e}".format(
                          i,
                          i + 1,
                          means_within_levels[i + 1],
                          rhs[i]
                        )
                    )
                ineq.append(False)

        # iterate over levels to check cross-level mean value decay
        for i in range(self.n_levels - 2):
            if rhs[i] > rhs[i+1]:
                decay.append(True)
            else:
                decay.append(False)

        try:
            if not np.array(ineq).all():
                msg = ("\nThe mean value inequality does not hold for all"
                       "two subsequent levels!")
                print(msg)
            else:
                print("\nThe mean value inequality holds between all two "
                      "subsequent levels!")
        
            if not np.array(decay).all():
                msg = ("The cross-level mean value does not decay "
                       "monotonically!")
                print(msg)
            else:
                print("The cross-level mean value decays monotonically!")
        
        except ValueError:
            if raise_error:
                raise
            else:
                pass

        return

    def calculate_thresholds(self):
        """ Calculate likelihood thresholds
        
        Calculate the thresholds according to the threshold fraction given
        by the user as a class attribute.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        thresholds = []
        for level in self.likelihoods_tuning:
            threshold = np.quantile(level, 1 - self.likelihood.threshold)
            thresholds.append(threshold)

        thresholds = np.asarray(thresholds)
        self.thresholds = thresholds
        print("\nThe calculated thresholds are: {}".format(self.thresholds))

        return
    
    def calculate_initial_bias_estimate_old(self):
        """ Calculate an intial estimate of the bias.

        Calculate the intial estimate of the bias from tuning samples. This
        results in a bias vector for each model level except for the
        highest-level model. I.e., the bias is estimated w.r.t. the
        highest-level model.

        Parameters
        ----------
        None

        Returns
        -------
        None        
        """

        # mu_k represents the bias w.r.t. the next higher level, computed
        # as the mean of the differences in results using all available
        # tuning results
        # initialize the data structure
        mu_k = []
        for k in range(self.n_levels - 1):
            mu_k_ = np.mean(
                (
                    self.results_analysis_tuning[k+1, :, :] -
                    self.results_analysis_tuning[k, :, :]
                ),
                axis=0
            )
            mu_k.append(mu_k_)
        
        # convert to numpy array
        mu_k = np.asarray(mu_k)

        # mu_B_l represents the total bias on any level w.r.t. the
        # highest-level model; it is the sum over bias on subsequent levels
        # starting from the lowest level
        # initialize data structure
        mu_B_l = []
        for l in range(self.n_levels - 1):
            mu_B_l.append(np.sum(mu_k[l:, :], axis=0))

        # append zeros for highest level as there is no bias
        mu_B_l.append(np.zeros_like(mu_k[0]))
        # convert to numpy array
        mu_B_l = np.asarray(mu_B_l)

        # set attribute
        self.bias = mu_B_l

        return
    
    def calculate_initial_bias_estimate(self):
        """Calculate an initial estimate of the bias with likelihood filtering.

        This function computes the initial bias estimates from tuning samples,
        considering only those samples where the likelihood is above the
        level-dependent threshold across all levels. The bias is estimated
        with respect to the highest-level model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Step 1: Create a mask for samples where all likelihoods are above thresholds
        # Shape of self.likelihoods_tuning: [n_levels, n_samples]
        # Shape of self.thresholds: [n_levels]
        # Broadcasting thresholds to match likelihoods shape for comparison
        likelihood_mask = self.likelihoods_tuning > self.thresholds[:, np.newaxis]
        
        # Combine masks across all levels using logical AND to ensure all are above thresholds
        # Resulting mask shape: [n_samples]
        valid_samples_mask = np.all(likelihood_mask, axis=0)
        
        # Debug: Print number of valid samples
        print(f"Number of valid samples after filtering: {np.sum(valid_samples_mask)} out of {self.likelihoods_tuning.shape[1]}")

        # Step 2: Apply the mask to filter results
        # Assuming self.results_analysis_tuning has shape [n_levels, n_samples, ...]
        # We need to preserve all dimensions after n_levels and n_samples
        # For example, if it's [n_levels, n_samples, n_features], we keep n_features
        # The filtered results will have shape [n_levels, n_valid_samples, ...]
        filtered_results = self.results_analysis_tuning[:, valid_samples_mask, ...]
        
        # Optional: Handle case with no valid samples
        if filtered_results.shape[1] == 0:
            raise ValueError("No samples passed the likelihood threshold filtering.")

        # Step 3: Calculate pairwise biases (mu_k) between adjacent levels
        mu_k = []
        for k in range(self.n_levels - 1):
            # Difference between level k+1 and level k
            diff = filtered_results[k+1, :, :] - filtered_results[k, :, :]
            
            # Compute mean across samples (axis=0)
            mu_k_ = np.mean(diff, axis=0)
            mu_k.append(mu_k_)

        # Convert mu_k list to a NumPy array
        mu_k = np.asarray(mu_k)  # Shape: [n_levels - 1, ...]
        
        # Step 4: Aggregate total bias relative to the highest level (mu_B_l)
        mu_B_l = []
        for l in range(self.n_levels - 1):
            # Sum biases from level l to the second-highest level
            # mu_k[l:, :] has shape [n_levels - 1 - l, ...]
            cumulative_bias = np.sum(mu_k[l:, :], axis=0)
            mu_B_l.append(cumulative_bias)

        # Append zeros for the highest level as it has no bias
        # Shape should match the bias vectors, e.g., [...], so use zeros_like
        highest_level_zero = np.zeros_like(mu_k[0])
        mu_B_l.append(highest_level_zero)
        
        # Convert mu_B_l list to a NumPy array
        mu_B_l = np.asarray(mu_B_l)  # Shape: [n_levels, ...]
        
        # Step 5: Assign the computed bias to the instance attribute
        self.bias = mu_B_l

        return

    
    def recalculate_likelihoods(self):
        """ Re-calculate likelihoods, accounting for bias.

        Re-calculate likelihoods while accounting for the bias from the
        initial estimate after tuning. During tuning, likelihoods are
        computed without accounting for bias. In order to include the bias-
        adapted likelihoods for hierarchy analysis and threshold
        computation, they need to be re-calculated.

        Parameters
        ----------
        None

        Returns
        -------
        None 
        """

        # initialize data structure
        likelihoods_re = []

        for level in range(self.n_levels - 1):
            likelihoods_re_level = []
            for i in range(self.results_analysis_tuning.shape[1]):
                # re-calculate likelihood
                lik = self.likelihood.likelihood(
                    obs=self.obs,
                    sim=self.results_analysis_tuning[level, i, :],
                    bias=self.bias[level, :]
                )
                likelihoods_re_level.append(lik)
            likelihoods_re.append(likelihoods_re_level)
        # append original likelihoods from highest level
        likelihoods_re.append(self.likelihoods_tuning[-1, :])

        likelihoods_re = np.asarray(likelihoods_re)
        if likelihoods_re.shape != self.likelihoods_tuning.shape:
            print(likelihoods_re.shape)
            print(self.likelihoods_tuning.shape)
            print("The shapes don't match...")

        self.likelihoods_tuning = likelihoods_re

        return
    
    def MLGLUE_sampling(self, samples):
        """Single-core MLGLUE sampling phase.

        Perform the sampling phase of MLGLUE without using multiprocessing
        / Ray.
        
        Parameters
        ----------
        samples : 2D list-like of float
            The parameter samples with which to perform sampling. Has to
            have the individual samples as rows and variables / model
            parameters as columns.

        Returns
        -------
        None

        Notes
        -----
        This is implemented as an instance method; it cannot be used
        independently of an MLGLUE instance. Therefore, results are not
        returned by this function but are rather stored in the class /
        instance attributes.
        """

        # set the internal level index to 0 (i.e., start sample evaluation
        # on level 0)
        run_id = 0

        # iterate over all samples
        for sample in samples:
            # evaluate the current sample
            sample_result = self.evaluate_sample(
                sample=sample,
                run_id=run_id
                )
            
            # if the sample is not finally accepted on the highest level,
            # continue to the next sample
            if sample_result[1] is False:
                # append the identifier for a highest-level model call
                self.highest_level_calls.append(sample_result[0])

                # increment the run_id
                run_id += 1
                continue
            
            # if the sample was accepted, handle the results returned above
            else:
                # the order of elements in the sample_result tuple is:
                # sample, likelihood_, results, results_analysis_sample,
                # highest_level_call
                self.selected_samples.append(sample_result[0])
                self.likelihoods.append(sample_result[1])
                self.results.append(sample_result[2])
                self.highest_level_calls.append(sample_result[4])

                # append level-dependent results to internal data structure
                for level_ in range(self.n_levels):
                    self.results_analysis[level_].append(
                        sample_result[3][level_]
                    )

                # increment the run_id
                run_id += 1

        return

    def evaluate_sample(self, sample, run_id):
        """Multiprocessing MLGLUE sampling utility.

        Evaluate a single parameter sample with the MLGLUE hierarchy. This
        design allows for multiprocessing (using Ray).
        
        Parameters
        ----------
        sample : 1D list-like of float
            The parameter sample which is evaluated. Each element has to
            represent an individual model parameter. Note that the order of
            the elements has to correspond to the order of model parameters
            in the model callable.
        run_id : int or str
            A run identifier in the form of an integer or string (the value
            of run_id is converted to a str later in any case). The value
            of run_id is also passed to the model callable, which is
            especially relevant if each model run is associated with a
            corresponding individual working directory. Then this directory
            can be named including the run_id value. This resolves problems
            for multiprocessing when multiple such individual directories
            are present in the same parent directory.
        
        Returns
        -------
        sample : 1D list-like of float
            The parameter sample which was evaluated.
        likelihood_ : float
            The likelihood value on the highest level corresponding to the
            parameter sample which was evaluated.
        results : 1D list-like of float
            The simulation results (at least all simulated observation
            equivalents) on the highest level corresponding to the sample
            which was evaluated.
        results_analysis_sample : 2D list-like of float
            The simulation results (at least all simulated observation
            equivalents) on all levels corresponding to the sample which
            was evaluated. Has the levels in rows and the individual
            results in columns.
        highest_level_call : int
            An identifier if the highest level model has been called (1) or
            not (0) using the evaluated parameter sample. It may happen
            that a parameter sample reaches the highest level model but is
            not accepted after running the model with that sample. Such
            cases should be minimized for MLGLUE to have optimal
            efficiency.

        Notes
        -----
        All returns are returned together as a tuple.
        """

        # set the call of the model on the highest level for the sample to
        # False (0)
        highest_level_call = 0

        # set the internal level index to 0 (i.e., start sample evaluation
        # on level 0)
        level_ = 0

        # initialize internal data structures
        results = None
        results_analysis_sample = []

        # evaluate the model on level 0 using the given parameter sample
        # and using the model callable from the corresponding instance
        # attribute
        results = self.model(
            parameters=sample,
            level=level_,
            n_levels=self.n_levels,
            run_id=run_id,
            )
        likelihood_ = self.likelihood.likelihood(
            obs=self.obs,
            sim=results,
            bias=self.bias[level_, :]
        )

        # if something went wrong (i.e., the model either returned None or
        # the initial value of results is still None), return None to
        # signal that evaluation for this sample was not successful. in
        # that case, the next sample is considered by the perform_MLGLUE
        # method.
        if results is None:
            return None

        # if results is not None (i.e., the model returned the expected
        # results), start passing the sample through the model hierarchy
        else:
            # append the results from the lowest level call to a list which
            # can be accessed by the user for further analysis
            results_analysis_sample.append(results)

            # initialize a variable to handle level indices
            level_checker = 0

            # iterate over the higher levels
            for level__ in range(1, self.n_levels):
                # check if the likelihood from the next coarser level is
                # above the corresponding level-dependent threshold
                level_checker = level__
                # if the likelihood is not None and is above the level-
                # dependent threshold, continue with the sample evaluation
                if likelihood_ is not None \
                    and likelihood_ >= self.thresholds[level__ - 1]:
                    
                    # if this already was a call to the model on the
                    # highest level, set the variable to 1 (True)
                    if level__ == self.n_levels - 1:
                        highest_level_call = 1
                    
                    # if the likelihood was above a threshold in the lower
                    # level, go up one level and compute the likelihood
                    # again
                    results = self.model(
                        parameters=sample,
                        level=level__,
                        n_levels=self.n_levels,
                        run_id=run_id,
                        )
                    likelihood_ = self.likelihood.likelihood(
                        obs=self.obs,
                        sim=results,
                        bias=self.bias[level__, :]
                    )

                    # append the model results to the analysis data
                    # structure
                    results_analysis_sample.append(results)

                # if the likelihood in the current level is below a
                # threshold, do not use the sample, break the level
                # iteration, and go to the next sample
                else:
                    # if a sample is not accepted on the highest level, the
                    # level_checker still is equal to the highest level
                    # index. in that case, it might happen below that
                    # unwillingly a sample results in a likelihood (from a
                    # lower level) above the threshold and the
                    # level_checker corresponding to the highest level. to
                    # circumvent this problem, the level_checker needs to
                    # be reduced by one
                    level_checker -= 1
                    break
            
            # if the likelihood is above the highest level threshold and
            # we are currently on the highest level, return all results
            if likelihood_ is not None \
                and likelihood_ >= self.thresholds[-1] \
                and level_checker == self.n_levels - 1:

                return (
                    sample,
                    likelihood_,
                    results,
                    results_analysis_sample,
                    highest_level_call
                    )

            # if the above conditions do not hold, return the highest-level
            # call variable and False to indicate that the next sample has
            # to be considered
            else:
                return (
                    highest_level_call,
                    False
                    )

    def evaluate_sample_tuning(self, sample, run_id):
        """Multiprocessing MLGLUE tuning utility.

        Evaluate a single parameter sample for tuning with the MLGLUE
        hierarchy. This design allows for multiprocessing (using Ray).
        Although similar to the evaluate_sample method, this method is
        slightly different: independently of the actual likelihood value,
        the sample is passed through all levels in the model hierarchy.
        
        Parameters
        ----------
        sample : 1D list-like of float
            The parameter sample which is evaluated. Each element has to
            represent an individual model parameter. Note that the order of
            the elements has to correspond to the order of model parameters
            in the model callable.
        run_id : int or str
            A run identifier in the form of an integer or string (the value
            of run_id is converted to a str later in any case). The value
            of run_id is also passed to the model callable, which is
            especially relevant if each model run is associated with a
            corresponding individual working directory. Then this directory
            can be named including the run_id value. This resolves problems
            for multiprocessing when multiple such individual directories
            are present in the same parent directory.
        
        Returns
        -------
        likelihoods_sample : 1D list-like of float
            The likelihood values on all levels corresponding to the
            parameter sample (except for tuning samples that result in an
            error or NaN returned by the model callable). The order is from
            the lowest to the highest level.
        results_analysis_tuning_sample : 2D list-like of float
            Holds simulated observation equivalents (ans possibly other
            quantities returned by the model) corresponding to the tuning
            sample (except for tuning samples that result in an error or
            NaN returned by the model callable) on all levels;
            has the tuning samples in the first dimension, the levels in
            the second dimension, and the simulated values in the third
            dimension.
        """

        # initialize internal data structures
        likelihoods_sample = []
        results_analysis_tuning_sample = []

        # set the internal level index to 0 (i.e., start sample evaluation
        # on level 0)
        level_ = 0

        # evaluate the model on level 0 using the given parameter sample
        # and using the model callable from the corresponding instance
        # attribute
        results = self.model(
            parameters=sample,
            level=level_,
            n_levels=self.n_levels,
            run_id=run_id,
            )
        likelihood_ = self.likelihood.likelihood(
            obs=self.obs,
            sim=results,
            bias=self.bias[level_, :]
        )

        # append likelihood value and results to internal data structures
        # the case where the likelihood is None etc. is handeled below
        likelihoods_sample.append(likelihood_)
        results_analysis_tuning_sample.append(results)

        # start passing the sample through the model hierarchy
        for level__ in range(1, self.n_levels):
            results = self.model(
                parameters=sample,
                level=level__,
                n_levels=self.n_levels,
                run_id=run_id,
                )
            likelihood_ = self.likelihood.likelihood(
                obs=self.obs,
                sim=results,
                bias=self.bias[level__, :]
            )

            # append likelihood value and results to internal data
            # structures the case where the likelihood is None etc. is
            # handeled below
            likelihoods_sample.append(likelihood_)
            results_analysis_tuning_sample.append(results)

        # handle the case where any likelihood is None in the model
        # hierarchy for the current sample. return noe to completely
        # discard all results renated to that sample
        if None in likelihoods_sample:
            return None
        
        # similarly check for NaN values and discard all results if found
        if np.isnan(likelihoods_sample).any():
            return None
        else:
            return likelihoods_sample, results_analysis_tuning_sample
        
    def check_samples(self):
        """Check user-specified parameter samples.

        Check the user-specified parameter samples. If a tuple is given,
        assign the elements to the respective attributes. If a single 2D
        array-like is given, split it into tuning and sampling parts.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if isinstance(self.samples, tuple):
            if self.multiprocessing == False:
                self.samples_tuning = self.samples[0]
                self.samples_sampling = self.samples[1]
            else:
                self.iterable_tuning = \
                    [(s, idx) for s, idx in zip(
                        self.samples[0],
                        [i for i in range(len(self.samples[0]))])]
                self.iterable_sampling = \
                    [(s, idx) for s, idx in zip(
                        self.samples[1],
                        [i + len(self.samples[0]) for i in range(
                            len(self.samples[1]))])]
        else:
            try:
                # get index at which to split the samples
                samples_divide = int(len(self.samples) * self.tuning)

                # split the samples
                if self.multiprocessing == False:
                    self.samples_tuning = self.samples[:samples_divide, :]
                    self.samples_sampling = self.samples[samples_divide:, :]
                else:
                    self.iterable_tuning = \
                        [(s, idx) for s, idx in zip(
                            self.samples[:samples_divide, :],
                            [i for i in range(
                                len(self.samples[:samples_divide, :]))])]
                    self.iterable_sampling = \
                        [(s, idx) for s, idx in zip(
                            self.samples[samples_divide:, :],
                            [i + len(self.samples[:samples_divide, :]) for \
                             i in range(len(self.samples[samples_divide:, :]))])]
            except:
                raise ValueError("The provided samples could not be "
                                 "split into tuning and sampling parts. "
                                 "Check the tuning fraction as well as "
                                 "the array of samples provided (samples "
                                 "have to have shape (n_samples, "
                                 "n_parameters)).")

        return
    
    def get_uniform_samples(self):
        """Get uniform parameter samples.

        Get uniform parameter samples in case no user-defined samples are
        provided. The user-defined upper and lower bounds of the parameter
        vector are used.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print("\nNo samples provided, using uniform sampling...")
        # generate samples
        self.samples = np.random.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            size=(int(self.n_samples), len(self.lower_bounds))
        )

        # split the samples into tuning and sampling parts
        self.check_samples()

        return
    
    def perform_MLGLUE_multiprocessing_tuning(self):
        """MLGLUE tuning using multiprocessing.

        Perform the MLGLUE tuning using multiprocessing / Ray. This only
        includes actual computations and not the preparation of samples
        etc.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # shut down ray and initialize again
        ray.shutdown()
        ray.init(num_cpus=self.n_processors)

        # perform tuning with multiprocessing
        print("\nStarting tuning with multiprocessing...")
        with Pool(processes=self.n_processors) as pool:
            for result in pool.starmap(self.evaluate_sample_tuning,
                                       self.iterable_tuning):
                if result is not None:
                    for num, i in enumerate(zip(result[0], result[1])):
                        self.likelihoods_tuning[num].append(i[0])
                        self.results_analysis_tuning[num].append(i[1])
        ray.shutdown()
        ray.shutdown()

        return

    def perform_MLGLUE_multiprocessing_sampling(self):
        """MLGLUE sampling using multiprocessing.

        Perform the MLGLUE sampling using multiprocessing / Ray. This only
        includes actual computations and not the preparation of samples
        etc.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        ray.shutdown()
        ray.init(num_cpus=self.n_processors)
        print("\nStarting sampling with multiprocessing...")
        with Pool(processes=self.n_processors) as pool:
            for eval_ in pool.starmap(self.evaluate_sample,
                                      self.iterable_sampling):
                if eval_ is not None and eval_[1] is not False:
                    self.selected_samples.append(eval_[0])
                    self.likelihoods.append(eval_[1])
                    self.results.append(eval_[2])
                    for num, i in enumerate(eval_[3]):
                        self.results_analysis[num].append(i)
                    if eval_[4] == 1:
                        self.highest_level_calls.append(eval_[4])
                elif eval_ is not None and eval_[1] is False:
                    if eval_[0] == 1:
                        self.highest_level_calls.append(eval_[0])
        ray.shutdown()

        return

    def perform_MLGLUE_singlecore_tuning(self):
        """MLGLUE tuning using a single CPU.

        Perform the MLGLUE tuning using a single CPU. This only includes
        actual computations and not the preparation of samples etc.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # perform tuning
        print("\nStarting tuning without multiprocessing...")
        self.MLGLUE_tuning(self.samples_tuning)

        return
    
    def perform_MLGLUE_singlecore_sampling(self):
        """MLGLUE sampling using a single CPU.

        Perform the MLGLUE sampling using a single CPU. This only includes
        actual computations and not the preparation of samples etc.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # perform sampling
        print("\nStarting sampling without multiprocessing...")
        self.MLGLUE_sampling(self.samples_sampling)

        return
    
    def perform_MLGLUE(self):
        """Perform the full MLGLUE algorithm. 
        
        Perform the full MLGLUE algorithm using single-core or parallelized
        computation. Settings for performing all computations are obtained
        from instance attributes.

        Parameters
        ----------
        None

        Returns
        -------
        selected_samples : 2D array-like of float
            The array of selected samples that are accepted on the highest
            level; has individual samples as rows and variables / model
            parameters as columns.
        likelihoods : 1D array-like of float
            The likelihood values correpsonding to the selected samples.
        results : 2D array-like of float
            Holds simulated observation equivalents corresponding to all
            posterior samples; has shape (len(selected_samples), len(obs)).

        Notes
        -----
        The returns of this function can always be accessed by referring to
        the correpsonding instance attributes. This function only returns
        those instance attributes for convenience.
        """
        # if multiprocessing is enabled, try to shut down any potentially
        # running Ray instance. if this throws an error (either because
        # there is nothing to shut down or because Ray is not installed),
        # the user is notified.
        if self.multiprocessing:
            try:
                ray.shutdown()
            except:
                print("\n-----")
                print("----")
                print("MLGLUE tried to shut down any running instances "
                      "of Ray. This resulted in an error. Please check "
                      "for any unwanted behaviour. MLGLUE will continue "
                      "now.")
                print("----")
                print("-----\n")

        # get the parameter samples
        try:
            if self.samples is None:
                self.get_uniform_samples()
            else:
                self.check_samples()
        except:
            raise ValueError("There was a problem obtaining the "
                             "parameter samples. Check the provided "
                             "samples, parameter bounds, n_samples, and "
                             "tuning fraction.")
        
        if self.multiprocessing:
            # perform tuning
            if self.thresholds_predefined == False:
                self.perform_MLGLUE_multiprocessing_tuning()

                # this is how it should be with the bias estimation:
                #   - standard tuning
                #   - standard hierarchy analysis
                #   - standard threshold computation
                #   - bias estimation only with samples for which the
                #       likelihood is ABOVE the level-dependent threshold
                #       (those can be different samples on different levels)
                #   - new hierarchy analysis (not necessary)
                #   - new threshold analysis
                #   - sampling

                # make results_analysis_tuning a numpy array
                self.results_analysis_tuning = np.asarray(
                    self.results_analysis_tuning
                )

                # make likelihoods_tuning a numpy array
                self.likelihoods_tuning = np.asarray(
                    self.likelihoods_tuning
                )

                # hierarchy analysis
                self.analyze_variances_likelihoods(
                    raise_error=self.hierarchy_analysis
                )
                self.analyze_means_likelihoods(
                    raise_error=self.hierarchy_analysis
                )

                # compute thresholds
                self.calculate_thresholds()

                if self.include_bias:
                    # if bias is included, compute the initial estimate
                    self.calculate_initial_bias_estimate()
                    
                    # we now need to re-calculate the likelihoods on all levels
                    # taking the bias into account
                    self.recalculate_likelihoods()

                    # new threshold analysis

            
            # perform sampling
            self.perform_MLGLUE_multiprocessing_sampling()
        
        else:
            # perform tuning
            if self.thresholds_predefined == False:
                self.perform_MLGLUE_singlecore_tuning()

                # make results_analysis_tuning a numpy array
                self.results_analysis_tuning = np.asarray(
                    self.results_analysis_tuning
                )

                # make likelihoods_tuning a numpy array
                self.likelihoods_tuning = np.asarray(
                    self.likelihoods_tuning
                )

                # hierarchy analysis
                self.analyze_variances_likelihoods(
                    raise_error=self.hierarchy_analysis
                )
                self.analyze_means_likelihoods(
                    raise_error=self.hierarchy_analysis
                )

                # compute thresholds
                self.calculate_thresholds()

                if self.include_bias:
                    # if bias is included, compute the initial estimate
                    self.calculate_initial_bias_estimate()
                    
                    # we now need to re-calculate the likelihoods on all levels
                    # taking the bias into account
                    self.recalculate_likelihoods()
            
            # perform sampling
            self.perform_MLGLUE_singlecore_sampling()

        print("\n\nSampling finished.\n\n")

        # convert data to arrays
        self.selected_samples = np.array(
            self.selected_samples
        )
        self.likelihoods = np.asarray(
            self.likelihoods
        )
        self.results = np.asarray(
            self.results
        )
        self.results_analysis = np.asarray(
            self.results_analysis
        )
        
        return (
            self.selected_samples,
            self.likelihoods,
            self.results
        )

    def estimate_uncertainty(self, quantiles=[0.01, 0.5, 0.99]):
        """Estimate simulation uncertainty.

        Estimate the simulation uncertainty, i.e., normalize likelihoods,
        and create probability densities for the model output. From that,
        uncertainty estimates given by user-defined quantiles are computed.
        The estimates are obtained by (1) normalizing likelihood values for
        individual values to always be in the range [0, 1], (2) ordering
        model outputs according to their associated normalized likelihood
        value, (3) estimating CDFs for the model outputs, and (4) computing
        quantile estimates.

        Parameters
        ----------
        quantiles : 1D list-like of float
            The quantiles with which to estimate the uncertainty of model
            outputs. Quantiles have to be given as floats in the range
            (0, 1). Default is [0.01, 0.50, 0.99].

        Returns
        -------
        uncertainty : 2D list-like of float
            The uncertainty estimates where individual model outputs are in
            rows and quantiles are in columns (in the order corresponding
            to the given quantiles).
        """

        # normalize likelihoods
        if quantiles is None:
            quantiles = [0.05, 0.5, 0.95]
        self.normalized_likelihoods = np.asarray(self.likelihoods) / np.sum(
            np.asarray(self.likelihoods))

        values = np.asarray(self.results)
        
        print("shape of values: ", np.shape(values))

        uncertainty = []

        # sort
        if values.ndim == 2:
            for i in range(np.shape(values)[1]):
                sorter = np.argsort(np.asarray(values[:, i]))
                values_ = values[:, i][sorter]
                weights = np.asarray(self.likelihoods)[sorter]

                # weighted_quantiles = np.cumsum(weights) - 0.5 * weights
                # weighted_quantiles /= np.sum(weights)

                weighted_quantiles = np.cumsum(weights) / np.sum(weights)

                uncertainty.append(np.interp(quantiles, weighted_quantiles,
                                             values_))

        elif values.ndim == 1:
            sorter = np.argsort(values)
            values_ = values[sorter]
            weights = np.asarray(self.likelihoods)[sorter]

            weighted_quantiles = np.cumsum(weights) - 0.5 * weights
            weighted_quantiles /= np.sum(weights)

            uncertainty.append(np.interp(quantiles, weighted_quantiles,
                                         values_))

        return np.asarray(uncertainty)

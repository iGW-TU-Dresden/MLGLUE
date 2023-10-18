import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import ray
from ray.util.multiprocessing import Pool

class MLGLUE():
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
            obs_x=None,
            obs_y=None,
            multiprocessing=False,
            n_processors=4,
            variance_analysis=None,
            savefigs="my_model",
            model_returns_all=False
    ):
        """
        :param model: the model for which parameters should be estimated (the
            model function should take the following arguments:
                parameters : an array of parameter values; 1D list-like
                level : an integer representing the current level index
                    (0-based); int
                n_levels : an integer representing the total number of
                    levels; int
                obs_x : coordinates of observed points (not directly used); 1D
                    list-like of floats or ints
                obs_y : observations in the same order as obs_x; 1D list-like of
                    floats or ints
                likelihood : the MLGLUE likelihood function (although user-defined
                    likelihood-functions may be used, the native MLGLUE functions
                    are tested); function
            and return a scalar value for the likelihood at the current parameter
            values; function
        :param likelihood: the likelihood to be used (this has to be a
            likelihood implemented in MLGLUE.Likelihoods); class
        :param lower_bounds: a 1D list-like specifying the lower bounds on the
            parameters; list-like
        :param upper_bounds: a 1D list-like specifying the upper bounds on the
            parameters; list-like
        :param n_samples: the number of samples to consider; int
        :param samples: a list-like containing the samples to consider, if samples
            is given, lower_bounds, upper_bounds, and n_samples are ignored;
            list-like
        :param tuning: the tuning fraction (int(n_samples * tuning) samples are
            used to analyze variances and to tune the likelihoodthreshold);
            0 < float < 1
        :param n_levels: the number of levels to consider; int
        :param coarsening_factor: the coarsening factor between levels (assuming
            geometric MLMC, may be defined as M_l = coarsening_factor * M_{l-1}
            for all l); int
        :param obs_x: a 1D list-like corresponding to the observation x-values;
            list-like
        :param obs_y: a 1D list-like corresponding to the observation y-values;
            list-like
        :param multiprocessing: whether to use multiprocessing or not; bool
        :param n_processors: number of processors to use if multiprocessing is
            True; int
        :param variance_analysis: variance analysis methodology after the
            tuning phase, valid options are "strong", "weak", None; str or None
        :param savefigs: whether to save variance analysis figures. if None,
            figures will not be saved, if str, figures will be saved with
            str as identifier. this parameter is ignored if variance_analysis
            is None; None or str
        :param model_returns_all: whether the model returns all simulation
            results besides the simulated equivalents of observations (i.e.,
            three returns instead of two returns by model); bool

        Note: if the model function only has one level, it should be the finest /
            target level.
        """
        self.model = model
        self.likelihood = likelihood
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.n_samples = n_samples
        self.samples = samples
        self.tuning = tuning
        self.thresholds = None
        self.n_levels = n_levels
        self.coarsening_factor = coarsening_factor
        self.obs_x = obs_x
        self.obs_y = obs_y
        self.multiprocessing = multiprocessing
        self.n_processors = n_processors
        self.variance_analysis = variance_analysis
        self.normalized_likelihoods = None
        self.savefigs = savefigs
        self.model_returns_all = model_returns_all

        # initialize output data structures
        self.results = [] # --> holds only the highest level model outputs
        self.selected_samples = [] # --> holds behavioural samples (samples that are accepted on the highest level)
        self.likelihoods = [] # --> holds highest level likelihood values corresponding to selected_samples
        self.likelihoods_tuning = [[] for i in range(self.n_levels)] # --> holds likelihood values for all levels from tuning
        self.results_analysis = [[] for i in range(self.n_levels)] # --> holds results from all levels during sampling
        self.results_analysis_tuning = [[] for i in range(self.n_levels)] # --> holds results from all levels during tuning
        self.thresholds = [] # --> holds likelihood thresholds on all levels
        self.full_results = [] # --> holds full model results (i.e., full array of heads for all model cells instead of
                                #   simulated equivalents of observed values)

        return

    def MLGLUE_tuning(self, samples):
        """
        Perform the tuning phase of MLGLUE

        :param samples: list-like with parameter samples for tuning with shape
            (n_samples, n_parameters); list-ike
        
        :return: likelihoods_tuning (the likelihoods corresponding to the
            samples)
        :return: results_analysis_tuning (the model outputs on each level for each sample)

        Note: if the model function only has one level, it should be the finest /
            target level.
        """
        run_id = 0
        for sample in samples:
            # start at the coarsest / lowest level
            level_ = 0

            # evaluate the model
            if self.model_returns_all == False:
                likelihood_, results = self.model(sample, level_, self.n_levels, self.obs_x,
                                                  self.obs_y, self.likelihood, run_id=run_id)
            else:
                likelihood_, results, results_full = self.model(sample, level_, self.n_levels, self.obs_x,
                                                                self.obs_y, self.likelihood, run_id=run_id)

            if results is None:
                run_id += 1
                continue

            else:
                self.likelihoods_tuning[level_].append(likelihood_)
                self.results_analysis_tuning[level_].append(results)

                # iterate over the higher levels
                for level__ in range(1, self.n_levels):
                    if self.model_returns_all == False:
                        likelihood_, results = self.model(sample, level__,
                                                          self.n_levels,
                                                          self.obs_x, self.obs_y,
                                                          self.likelihood,
                                                          run_id=run_id)
                    else:
                        likelihood_, results, results_full = self.model(sample, level__,
                                                                        self.n_levels,
                                                                        self.obs_x, self.obs_y,
                                                                        self.likelihood,
                                                                        run_id=run_id)

                    self.likelihoods_tuning[level__].append(likelihood_)
                    self.results_analysis_tuning[level__].append(results)

                # check that no nan values are present in the likelihoods
                # if yes, remove all likelihoods
                if np.nan in [col[-1] for col in self.likelihoods_tuning]:
                    for i in self.likelihoods_tuning:
                        i.pop()

                if None in [col[-1] for col in self.likelihoods_tuning]:
                    for i in self.likelihoods_tuning:
                        i.pop()

                run_id += 1

        return self.likelihoods_tuning, self.results_analysis_tuning

    def analyze_variances_likelihoods_strong(self):
        """
        Analyze the relationships between levels in terms of variances:
            - variances of likelihood values on each individual level
            - covariances of likelihoods across levels
            - variances of the difference between likelihood values on subsequent levels

        Specifically, the following inequality is evaluated for all levels:
            Var[L_(l) - L_(l-1)] = Var[L_(l)] + Var[L_(l-1)] - 2 * Cov[L_(l), L_(l-1)]

        MLGLUE is stopped if 2 * Cov[L_(l), L_(l-1)] < Var[L_(l-1)] for any l

        :return: None
        """

        self.likelihoods_tuning = np.asarray(self.likelihoods_tuning)

        # get number of likelihoods in every level
        # n_vals = len(self.likelihoods_tuning[0, :])
        lens = []
        for i in range(self.n_levels):
            lens.append(len(self.likelihoods_tuning[i]))
        n_vals = min(lens)

        # vars_within_levels contains the variance at every x-value for each
        #   level (n_levels, n_steps)
        vars_within_levels = []
        for i in self.likelihoods_tuning:
            # calculate the variance within the level
            var_ = np.var(i, axis=0, ddof=1)
            # append variance to list
            vars_within_levels.append(var_)

        # calculate covariance between / across levels (0 - 1), (1 - 2), ...
        covs_cross_levels = []
        for i in range(self.n_levels - 1):
            cov = np.cov(self.likelihoods_tuning[i + 1][:n_vals],
                         self.likelihoods_tuning[i][:n_vals],
                         ddof=1)[0, 1]
            covs_cross_levels.append(cov)

        # calculate right hand side of inequality
        rhs = []
        for i in range(self.n_levels - 1):
            rhs_ = (vars_within_levels[i] + vars_within_levels[i + 1] - 2. *
                    covs_cross_levels[i])
            rhs.append(rhs_)

        # calculate logarithms of data
        vars_within_levels = np.log(vars_within_levels) / np.log(self.coarsening_factor)
        rhs = np.log(rhs) / np.log(self.coarsening_factor)

        # initialize list to store booleans representing whether the veriance
        #   inequality holds or not
        ineq = []
        # initialize list to store booleans representing whether the cross-level
        #   variance decays
        decay = []
        print("Covariances across levels: ", covs_cross_levels)
        print("Variances within levels: ", vars_within_levels)
        print("Variances across levels: ", rhs)
        # iterate over levels to check variance inequality
        for i in range(self.n_levels - 1):
            if vars_within_levels[i + 1] >= rhs[i]:
                print("The variance inequality holds between levels ",
                      "{} and {}: {:1.5f} >= {:1.5e}".format(i, i + 1,
                                                             vars_within_levels[i + 1],
                                                             rhs[i]))
                ineq.append(True)
            else:
                print("The variance inequality does NOT hold between levels ",
                      "{} and {}: {:1.5f} >//= {:1.5e}".format(i, i + 1,
                                                               vars_within_levels[i + 1],
                                                               rhs[i]))
                ineq.append(False)

        # iterate over levels to check cross-level variance decay
        for i in range(self.n_levels - 2):
            if rhs[i] > rhs[i+1]:
                decay.append(True)
            else:
                decay.append(False)

        # throw error if the inequality does not hold for one level combination
        try:
            if not np.array(ineq).all():
                msg = ("The variance inequality does not hold for all two "
                       "subsequent levels!")
                print(msg)
                raise ValueError(msg)
            else:
                print("The variance inequality holds between all two "
                      "subsequent levels!")

            if not np.array(decay).all():
                msg = ("The cross-level variance does not decay monotonically!")
                print(msg)
                raise ValueError(msg)
            else:
                print("The cross-level variance decays monotonically!")

        except ValueError:
            raise

        return

    def analyze_means_likelihoods_strong(self):
        """
        Analyze the relationships between levels in terms of mean values:
            - means of likelihood values on each individual level
            - means of the difference between likelihood values on subsequent levels

        :return: None
        """

        self.likelihoods_tuning = np.asarray(self.likelihoods_tuning)

        # get number of likelihoods in every level
        # n_vals = len(self.likelihoods_tuning[0, :])
        lens = []
        for i in range(self.n_levels):
            lens.append(len(self.likelihoods_tuning[i]))
        n_vals = min(lens)

        means_within_levels = []
        for i in self.likelihoods_tuning:
            # calculate the variance within the level
            mean_ = abs(np.mean(i, axis=0))
            # append variance to list
            means_within_levels.append(mean_)

        # calculate mean of difference
        rhs = []
        for i in range(self.n_levels - 1):
            rhs_ = abs(means_within_levels[i] - means_within_levels[i + 1])
            rhs.append(rhs_)

        # compute logarithms of data
        # apparently, a coarsening factor is in the exponent --> for a rectangular mesh,
        #   doubling the number of cells in each direction correpsonds to a coarsening
        #   factor of 2
        means_within_levels = np.log(means_within_levels) / np.log(self.coarsening_factor)
        rhs = np.log(rhs) / np.log(self.coarsening_factor)

        ineq = []
        print("Means within levels: ", means_within_levels)
        print("Means values of difference between levels: ", rhs)

        return

    def analyze_variances_likelihoods_weak(self):
        """
        Analyze the relationships between levels in terms of the correlation between
        subsequent levels.

        MLGLUE is stopped if there is zero or negative correlation between
        subsequent levels.

        :return: None
        """

        self.likelihoods_tuning = np.asarray(self.likelihoods_tuning)

        # get number of likelihoods in every level
        n_vals = len(self.likelihoods_tuning[0, :])

        # calculate correlation between / across levels (0 - 1), (1 - 2), ...
        corrs_cross_levels = []
        for i in range(self.n_levels - 1):
            corr = np.corrcoef(self.likelihoods_tuning[i][:n_vals],
                               self.likelihoods_tuning[i + 1][:n_vals])[0, 1]
            corrs_cross_levels.append(corr)

        ineq = []
        print("Correlations across levels: ", corrs_cross_levels)
        for i in range(self.n_levels - 1):
            if corrs_cross_levels[i] > 0:
                print("The correlation inequality holds between levels ",
                      "{} and {}: {:1.5f} > 0".format(i, i + 1,
                                                      corrs_cross_levels[i]))
                ineq.append(True)
            else:
                print("The correlation inequality does NOT hold between "
                      "levels ",
                      "{} and {}: {:1.5f} <= 0".format(i, i + 1,
                                                       corrs_cross_levels[i]))
                ineq.append(False)

        # throw error if the inequality does not hold for one level combination
        try:
            if not np.array(ineq).all():
                msg = ("The correlation inequality does not hold for all two "
                       "subsequent levels!")
                raise ValueError(msg)
            else:
                print("The correlation inequality holds between all two "
                      "subsequent levels! Everything is fine!")
        except ValueError:
            raise

        return

    def calculate_threshold(self):
        """
        Calculate the thresholds according to the threshold fraction given by
        the likelihood function

        :return: thresholds (list-like specifying the float threshold values
            on each level)
        """

        counter = 0
        for level in self.likelihoods_tuning:
            threshold = np.quantile(level, 1 - self.likelihood.threshold)
            self.thresholds.append(threshold)
            counter += 1

        # get the maximum threshold and use that for all levels
        # self.thresholds = [max(self.thresholds)] * self.n_levels

        print("\nThe calculated thresholds are: {}".format(self.thresholds))

        return self.thresholds

    def MLGLUE_sampling(self, samples):
        """
        Perform the sampling phase of MLGLUE

        :param samples: list-like of parameter samples to evaluate with shape
            (n_samples, n_parameters); list-like

        :return: selected_samples, list-like of behavioural parameter samples
        :return: likelihoods, list-like of likelihood values corresponding to the
            selected_samples

        Note: if the model function only has one level, it should be the finest /
            target level.
        """

        run_id = 0
        for sample in samples:
            # start at the coarsest / lowest level
            level_ = 0

            # initialize results
            results = None

            # evaluate the model
            if self.model_returns_all == False:
                likelihood_, results = self.model(sample, level_, self.n_levels, self.obs_x,
                                                  self.obs_y, self.likelihood, run_id=run_id)
            else:
                likelihood_, results, results_full = self.model(sample, level_, self.n_levels, self.obs_x,
                                                                self.obs_y, self.likelihood, run_id=run_id)

            if results is None:
                continue

            else:
                self.results_analysis[level_].append(results)

                level_checker = 0
                # iterate over the higher levels
                for level__ in range(1, self.n_levels):
                    # check if the likelihood from the next coarser level is above a
                    #     certain threshold
                    level_checker = level__
                    if likelihood_ is not None and likelihood_ >= self.thresholds[level__ - 1]:
                        # if the likelihood was above a threshold in the lower
                        #   level, go up one level and compute the likelihood again
                        if self.model_returns_all == False:
                            likelihood_, results = self.model(sample, level__,
                                                              self.n_levels,
                                                              self.obs_x,
                                                              self.obs_y,
                                                              self.likelihood,
                                                              run_id=run_id)
                        else:
                            likelihood_, results, results_full = self.model(sample, level__,
                                                                            self.n_levels,
                                                                            self.obs_x,
                                                                            self.obs_y,
                                                                            self.likelihood,
                                                                            run_id=run_id)

                        self.results_analysis[level__].append(results)

                    # if the likelihood in the current level is below a
                    #   threshold, do not use the sample, break the level
                    #   iteration, and go to the next sample
                    else:
                        break

                if (likelihood_ is not None and
                        likelihood_ >= self.thresholds[-1] and
                        level_checker == self.n_levels - 1):
                    if self.model_returns_all == True:
                        self.full_results.append(results_full)
                    self.selected_samples.append(sample)
                    self.likelihoods.append(likelihood_)
                    self.results.append(results)

                run_id += 1

        # the number of selected samples is equal to the number of model results
        #     for the highest level
        return self.selected_samples, self.likelihoods

    # @ray.remote
    def evaluate_sample(self, sample, run_id):
        """
        A function for evaluate a sample that can be used in a multiprocessing
        framework; used during the sampling phase

        :param sample: the current sample; list-like
        :param run_id: a run identifier; string or int
        :return: a tuple of (selected_sample, likelihood, results, results_analysis_sample, results_full),
            if the sample is accepted; None is returned if the sample is not accepted

        Note: the selected sample as well as the corresponding likelihood are
            appended to a global list that is returned by the perform_MLGLUE
            method
        """

        # start at the coarsest / lowest level
        level_ = 0

        # initialize results
        results = None
        results_analysis_sample = []

        # evaluate the model
        if self.model_returns_all == False:
            likelihood_, results = self.model(sample, level_, self.n_levels, self.obs_x,
                                              self.obs_y, self.likelihood, run_id=run_id)
        else:
            likelihood_, results, results_full = self.model(sample, level_, self.n_levels, self.obs_x,
                                                            self.obs_y, self.likelihood, run_id=run_id)

        if results is None:
            return None

        else:
            results_analysis_sample.append(results)

            level_checker = 0
            # iterate over the higher levels
            for level__ in range(1, self.n_levels):
                # check if the likelihood from the next coarser level is above a
                #     certain threshold
                level_checker = level__
                if likelihood_ is not None and likelihood_ >= self.thresholds[level__ - 1]:
                    if level__ == self.n_levels - 1:
                        print("highest level call: ", run_id)
                    # if the likelihood was above a threshold in the lower
                    #   level, go up one level and compute the likelihood again
                    if self.model_returns_all == False:
                        likelihood_, results = self.model(sample, level__,
                                                          self.n_levels,
                                                          self.obs_x, self.obs_y,
                                                          self.likelihood,
                                                          run_id=run_id)
                    else:
                        likelihood_, results, results_full = self.model(sample, level__,
                                                             self.n_levels,
                                                             self.obs_x, self.obs_y,
                                                             self.likelihood,
                                                             run_id=run_id)

                    results_analysis_sample.append(results)

                # if the likelihood in the current level is below a
                #   threshold, do not use the sample, break the level
                #   iteration, and go to the next sample
                else:
                    # if a sample is not accepted on the highest level, the level_checker
                    #   still is equal to the highest level index. In that case, it might
                    #   happen below that unwillingly a sample results in a likelihood
                    #   (from a lower level) above the threshold and the level_checker
                    #   corresponding to the highest level. To circumvent this problem,
                    #   the level_checker needs to be reduced by one
                    level_checker -= 1
                    break

            if (likelihood_ is not None and
                    likelihood_ >= self.thresholds[-1] and
                    level_checker == self.n_levels - 1):
                if self.model_returns_all == True:
                    # append full results
                    return (sample, likelihood_, results, results_analysis_sample, results_full)
                else:
                    return (sample, likelihood_, results, results_analysis_sample)
            else:
                return None

    # @ray.remote
    def evaluate_sample_tuning(self, sample, run_id):
        """
        A function for evaluating a sample that can be used in a multiprocessing
        framework; used during the tuning phase

        :param sample: the current sample; list-like
        :return: a tuple of (selected_sample, likelihood), if the sample is
            accepted; None is returned if the sample is not accepted
        """

        likelihoods_sample = []
        results_analysis_tuning_sample = []

        # start at the coarsest / lowest level
        level_ = 0

        # evaluate the model
        if self.model_returns_all == False:
            likelihood_, results = self.model(sample, level_, self.n_levels, self.obs_x,
                                              self.obs_y, self.likelihood, run_id=run_id)
        else:
            likelihood_, results, results_full = self.model(sample, level_, self.n_levels, self.obs_x,
                                                            self.obs_y, self.likelihood, run_id=run_id)

        likelihoods_sample.append(likelihood_)
        results_analysis_tuning_sample.append(results)

        # iterate over the higher levels
        for level__ in range(1, self.n_levels):
            if self.model_returns_all == False:
                likelihood_, results = self.model(sample, level__,
                                                  self.n_levels,
                                                  self.obs_x, self.obs_y,
                                                  self.likelihood, run_id=run_id)
            else:
                likelihood_, results, results_full = self.model(sample, level__,
                                                                self.n_levels,
                                                                self.obs_x, self.obs_y,
                                                                self.likelihood, run_id=run_id)

            likelihoods_sample.append(likelihood_)
            results_analysis_tuning_sample.append(results)

        if None in likelihoods_sample:
            return None
        # check that no nan values are present in the likelihoods
        # if yes, remove all likelihoods
        if np.isnan(likelihoods_sample).any():
            return None
        else:
            return likelihoods_sample, results_analysis_tuning_sample

    def perform_MLGLUE(self):
        """
        Perform the full MLGLUE, including a tuning phase where the
        inequality regarding variances is checked

        :return: selected_samples; list-like of behavioural samples
        :return: likelihoods; list-like of likelihoods corresponding to the behavioural samples
        :return: results; list-like of model outputs on all levels for the behavioural samples
        :return: results_full; list-like of full model results (optional)
        """

        if self.samples is None:
            print("No samples provided, using uniform sampling...")
            # generate samples
            # np.random.seed(2)
            # np.random.seed(42)
            self.samples = np.random.uniform(
                low=self.lower_bounds,
                high=self.upper_bounds,
                size=(int(self.n_samples), len(self.lower_bounds))
            )

        else:
            self.n_samples = len(self.samples)

        # get index at which to split the samples
        samples_divide = int(self.n_samples * self.tuning)

        # split the samples
        samples_tuning = self.samples[:samples_divide, :]
        samples_sampling = self.samples[samples_divide:, :]

        # make iterable of iterables for starmap
        # each inner iterable (tuple) contains the current sample as well as a
        #   sample index
        if self.multiprocessing:
            iterable_tuning = [(s, idx) for s, idx in zip(samples_tuning,
                                                          [i for i in range(samples_divide)])]
            iterable_sampling = [(s, idx) for s, idx in zip(samples_sampling,
                                                            [i + samples_divide for i in range(self.n_samples - samples_divide)])]

        if not self.multiprocessing:
            # perform tuning
            print("\nStarting tuning without multiprocessing...")
            _ = self.MLGLUE_tuning(samples_tuning)

        elif self.multiprocessing:
            ray.shutdown()
            ray.init(num_cpus=self.n_processors)
            # perform tuning with multiprocessing
            print("\nStarting tuning with multiprocessing...")
            with Pool(processes=self.n_processors) as pool: # map # maxtasksperchild=1
                for result in pool.starmap(self.evaluate_sample_tuning, iterable_tuning):
                    if result is not None:
                        for num, i in enumerate(zip(result[0], result[1])):
                            self.likelihoods_tuning[num].append(i[0])
                            self.results_analysis_tuning[num].append(i[1])
            ray.shutdown()
            ray.shutdown()

        # analyze variances and mean values
        if self.variance_analysis == "strong":
            self.analyze_variances_likelihoods_strong()
            self.analyze_means_likelihoods_strong()
        elif self.variance_analysis == "weak":
            self.analyze_variances_likelihoods_weak()
            self.analyze_means_likelihoods_strong()
        elif self.variance_analysis is None:
            pass
        else:
            print("No valid variance analysis methodology selected; "
                  "continuing without analysis!")

        # compute thresholds
        self.thresholds = self.calculate_threshold()
        # self.thresholds = [0., 0., 0.]

        if not self.multiprocessing:
            # perform sampling
            print("\nStarting sampling without multiprocessing...")
            selected_samples, likelihoods = self.MLGLUE_sampling(
                samples_sampling)

            if self.model_returns_all == False:
                return selected_samples, likelihoods, self.results
            else:
                return selected_samples, likelihoods, self.results, self.full_results

        elif self.multiprocessing:
            ray.shutdown()
            ray.init(num_cpus=self.n_processors)
            print("\nStarting sampling with multiprocessing...")
            with Pool(processes=self.n_processors) as pool: # maxtasksperchild=1
                for eval_ in pool.starmap(self.evaluate_sample, iterable_sampling):
                    # TODO: this has to be improved somehow; sometimes,
                    #  eval_[2] is None here but the result for the model
                    #  call is not actually None
                    if eval_ is not None and eval_[2] is not None:
                        self.selected_samples.append(eval_[0])
                        self.likelihoods.append(eval_[1])
                        self.results.append(eval_[2])
                        for num, i in enumerate(eval_[3]):
                            self.results_analysis[num].append(i)
                        if self.model_returns_all == True:
                            self.full_results.append(eval_[4])
            ray.shutdown()

            if self.model_returns_all == False:
                return self.selected_samples, self.likelihoods, self.results
            else:
                return self.selected_samples, self.likelihoods, self.results, self.full_results

    def estimate_uncertainty(self, quantiles=None):
        """
        Estimate the simulation uncertainty, i.e., normalize likelihoods,
        and create a probability density for the model output

        :param quantiles: the quantiles to return for the prediction;
            list-like of three float values [lower q., 0.5, upper q.]
        :return: uncertainty; list-like with shape (n_outputs, 3) with the
            uncertainty estimates
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

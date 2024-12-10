The Concept of MLGLUE
=====================

In a nutshell, MLGLUE can be used to accelerate Bayesian inference of
(hydrological) model parameters. We assume that you are familiar with
the general concept of Bayesian inference. For more details, references,
etc., please also look at the original publication about MLGLUE
`here <https://doi.org/10.1029/2024WR037735>`_.

Inverse problems and Bayesian inference
---------------------------------------
We consider a model :math:`\mathcal{F}(\cdot)`, which simulates observed
values :math:`\mathbf{d}` (up to errors :math:`\boldsymbol \varepsilon`),
using model parameters :math:`\boldsymbol \theta`:

.. math::
	\mathbf{d} = \mathcal{F}(\boldsymbol \theta) + \boldsymbol \varepsilon

Our aim when solving an inverse problem is to find parameters of the model
such that the model simulations match the corresponding observations as
closely as possible.

We consider :math:`\boldsymbol \theta` to be a vector of random variables,
which is associated with a prior probability distribution,
:math:`p_{prior}(\boldsymbol \theta)`. Conditioning the prior on
observations leads to the posterior probability distribution of the
parameters, :math:`p_{post}(\boldsymbol \theta | \mathbf{d})` via Bayes'
theorem:

.. math::
	p_{post}\left(\boldsymbol \theta | \mathbf{d}\right) \propto
	p_{prior}\left(\boldsymbol \theta\right)
	\mathcal{L}\left(\boldsymbol \theta | \mathbf{d}\right)

Here, :math:`\mathcal{L}\left(\boldsymbol \theta | \mathbf{d}\right)` is
the *likelihood*.

While the Bayesian approach is rather intuitive, we can usually not obtain
the posterior analytically - we have to generate samples from the posterior
by computing the likelihood for many different samples of
:math:`\boldsymbol \theta` and each sample requires a model evaluation.
Now if each model run is computationally costly, this approach quickly
becomes intractable. Multilevel methods can help to alleviate the
computational burden of the problem to allow for sampling-based approaches
to Bayesian inference with costly models.

.. _multilevel methods:

Multilevel methods
------------------
The central idea of multilevel methods is simple: instead of computing a
Monte Carlo estimate of a quantity of interest (QoI) using a model with
high resolution, high accuracy and high computational cost we rely on
models with lower resolution, lower accuracy, and lower computational cost
to do most of the work.
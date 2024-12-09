The Concept of MLGLUE
=====================

In a nutshell, MLGLUE can be used to accelerate Bayesian inference of
(hydrological) model parameters.
In the following sections, general aspects about Bayesian inference as well
as the details about MLGLUE are explained. Plase also look at the original
publication about MLGLUE `here <https://doi.org/10.1029/2024WR037735>`_.

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

.. admonition:: Example
	
	Let's consider an example we can re-use later: we assume the model to
	simulate steady-state flow in a porous medium. Our observed data
	consists of pressure measurements at different locations. The only
	model parameter we consider is the homogeneous macroscopic permeability
	of the porous medium.

There are many different approaches to solving such problems. Bayesian
inference is a highly popular statistical approach to the problem with many
different specific methods associated with it. As a statistical approach it
can be summarized as follows: we first consider all of our model parameters
to be random variables. Without taking observations into account we
define a *prior* distribution (or just *prior* for short) of the model
parameters. Finally we condition the *prior* on our observations to obtain
a *posterior* distribution (or just *posterior* for short). This
conditioning is formalized in **Bayes' theorem**:

.. math::
	p_{post}\left(\boldsymbol \theta | \mathbf{d}\right) \propto
	p_{prior}\left(\boldsymbol \theta\right)
	\mathcal{L}\left(\boldsymbol \theta | \mathbf{d}\right)

Here, :math:`\mathcal{L}\left(\boldsymbol \theta | \mathbf{d}\right)` is
the *likelihood*, which can be understood of a way to assess the
goodness-of-fit of a certain value of :math:`\boldsymbol \theta`.
Evaluating :math:`\mathcal{L}` requires running our model to obtain
simulated values which correspond to the specific
:math:`\boldsymbol \theta` we are considering.

.. admonition:: Example

	Coming back to the example, we can think of the conditioning step as a
	form of *updating*. The prior of our permeability parameter may be very
	wide; without any observations we just don't know anything about the
	permeability. However, if observations become available, they can
	constrain our prior beliefs and we update the *prior* using
	Bayes' theorem to obtain the *posterior*.

While the Bayesian approach is rather intuitive, we can usually not obtain
the posterior analytically - we have to generate samples from the posterior
by computing the likelihood for many differen samples of
:math:`\boldsymbol \theta`. As explained before, each sample requires a
model evaluation. Now if each model run is computationally costly, this
approach quickly becomes intractable.

.. admonition:: Example

	Imagine our example model to have a run-time of hours, days or even
	weeks. This is common for complex models which are often used in
	practice. Even if we could run multiple instances of our model in
	parallel, pushing thousand or millions of parameter samples through the
	model will obviously not work in a practical context.

.. caution::
	There are many details, pitfalls, and caveats we did not mention in the
	paragraphs above. But this is the general setting of Bayesian inference and
	how computational cost can decide upon the applicability of such an
	approach.

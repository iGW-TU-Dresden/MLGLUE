import numpy as np

"""
Here we develop and test the utilities for supporting a bias term in the
likelihood function. This then results in a level-dependent likelihood.
Similar to the Adaptive Error Model by Lykkegaard et al., this helps
to work with models on coarse levels which show bias relative to the
highest-level model.
We get an initial estimate for the bias after the tuning phase. We then
perform all analysis and the computation of thresholds after we adapt
the likelihoods with the bias.
"""

def calculate_initial_bias_estimate():
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

    results_analysis_tuning = np.array([
        [
            [40., 60.],
            [45., 55.],
            [50., 50.]
        ],
        [
            [60., 80.],
            [65., 75.],
            [70., 70.]
        ],
        [
            [70., 90.],
            [75., 85.],
            [80., 80.]
        ]
    ])

    levels = results_analysis_tuning.shape[0]

    mu_k = []
    for k in range(levels - 1):
        mu_k_ = np.mean(
            (
                results_analysis_tuning[k+1, :, :] - 
                results_analysis_tuning[k, :, :]
            ),
            axis=0
        )
        mu_k.append(mu_k_)
    mu_k = np.asarray(mu_k)

    mu_B_l = []
    for l in range(levels - 1):
        mu_B_l.append(np.sum(mu_k[l:, :], axis=0))

    # append zeros for highest level as there is no bias
    mu_B_l.append(np.zeros_like(mu_k[0]))
    mu_B_l = np.asarray(mu_B_l)

    print(mu_B_l)

    return

_ = calculate_initial_bias_estimate()
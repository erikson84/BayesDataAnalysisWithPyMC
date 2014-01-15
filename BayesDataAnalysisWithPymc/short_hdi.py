# -*- coding: utf-8 -*-
'''Algorithm to calculate the shortest Highest Density Interval
(HDI). Adaptation of the R code from "Doing Bayesian Data Analysis",
by John K. Krushcke.
More info: http://doingbayesiandataanalysis.blogspot.com.br/

'''


def short_hdi(sample, cred=0.95):
    '''Calculate the shortest Highest Density Interval from
    the posterior distribution sampled via MCMC.

    :Arguments:
        sample: A list with the values of the posterior distribution.
        cred: The mass of the posterior for which the interval is computed.
                Default is 95%, should be a float from 0.0 to 1.0.

    Returns a tuple with the limits of the HDI.

    PyMC has a 95% HDI algorithm, but it uses quantiles.

    '''
    sorted_sample = sorted(sample)
    ci_index = int(cred * len(sorted_sample))  # Uses 'int()' for R's 'floor()'
    num_ci = len(sorted_sample) - ci_index

    ci_width = []
    for i in range(num_ci):
        width = sorted_sample[i + ci_index] - sorted_sample[i]
        ci_width.append(width)

    hdi_min = sorted_sample[ci_width.index(min(ci_width))]
    hdi_max = sorted_sample[ci_width.index(min(ci_width)) + ci_index]
    hdi_lim = (hdi_min, hdi_max)
    return hdi_lim

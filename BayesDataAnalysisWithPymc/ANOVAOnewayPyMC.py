# -*- coding: utf-8 -*-
'''Hierarchical Model for estimation of oneway ANOVA parameters via MCMC.
Python (PyMC) adaptation of the R code from "Doing Bayesian Data Analysis",
by John K. Krushcke.
More info: http://doingbayesiandataanalysis.blogspot.com.br/

'''
from __future__ import division

import pymc
import numpy as np
from matplotlib import pyplot as plot
from plot_post import plot_post
from normalize import *
from math import ceil
from os import path

# Code to find the data path.

scr_dir = path.dirname(__file__)
file_name = 'McDonaldSK1991data.txt'
comp_dir = path.join(scr_dir, 'Data', file_name)

# Using data from the book for easier comparison.
# Data from McDonald (1991) study about geographical location and muscle size
# in mussels.
# Again we use Numpy to assign the data to variables.

x, y = np.genfromtxt(comp_dir, delimiter=' ',
                     skip_header=19, usecols=(0, 1), unpack=True)

# Define the contrasts.
# TODO: use dictionary for easier retrieval of contrast description.

contrasts = np.array(((-1/3, -1/3, 1/2, -1/3, 1/2),
                      (1, -1, 0, 0, 0),
                      (-1/2, -1/2, 1, 0, 0),
                      (-1/2, -1/2, 1/2, 1/2, 0),
                      (1/3, 1/3, 1/3, -1, 0),
                      (-1/4, -1/4, -1/4, -1/4, 1),
                      (1/3, 1/3, 1/3, -1/2, -1/2),
                      (0, 0, 0, -1, 1)))
# Random data, for test purposes:

# y_truesd = 4.0
# a0_true = 100
# atrue = [15, -10, -7, 8, -6]

#x = [1] * 3 + [2] * 4 + [3] * 3 + [4] * 5 + [5] * 3
#y = [a0_true + atrue[i - 1] + np.random.normal(0, y_truesd) for i in x]


# Normalize the data for better MCMC performance.
# And define the total number of levels in our categorical variable.

zy = normalize(y)
x_levels = len(set(x))
y_mean = np.mean(y)
y_sd = np.sqrt(np.var(y))

# Begin the definition of the model.
# First, we define a Gamma distribution for the precision of
# the deflection parameters.

a_sd = pymc.Gamma('a_sd', 1.01005, 0.1005)

@pymc.deterministic
def a_tau(a_sd=a_sd):
    return 1.0 / a_sd**2

# Then we define a normal prior on the baseline and deflection parameters.

a0 = pymc.Normal('a0', mu=0.0, tau=0.001)
a = pymc.Normal('a', mu=0.0, tau=a_tau, size=x_levels)

# Almost there! We still need to set the prior on the data variance.

sigma = pymc.Uniform('sigma', 0, 10)


@pymc.deterministic
def tau(sigma=sigma):
    return 1.0 / sigma**2

# The priors are all set! Now we can define the linear model.
# Maybe it can be clearly defined using the 'Lambda()' class.
# But we will use a 'for' loop for easier readability.

mu = []
for i in x:
    mu.append(a0 + a[int(i - 1)])

# And the likelihood.

like_y = pymc.Normal('like_y', mu=mu, tau=tau, value=zy, observed=True)

# Now we build the model, set the MAP and sample the posterior distribution.

model = pymc.Model([like_y, a0, a, sigma, a_tau, a_sd])
map_ = pymc.MAP(model)
map_.fit()
mcmc = pymc.MCMC(model)
mcmc.sample(iter=80000, burn=20000, thin=10)

# Extract the samples.

a0_sample = mcmc.trace('a0')[:]
a_sample = mcmc.trace('a')[:]
sigma_sample = mcmc.trace('sigma')[:]
a_sd_sample = mcmc.trace('a_sd')[:]

# Convert the values.

#m_sample = a0_sample.repeat(x_levels).reshape(len(a0_sample), x_levels) + a_sample

#b0_sample = m_sample.mean(axis=1)
b0_sample = convert_baseline(a0_sample, a_sample, x_levels, y)

#b_sample = (m_sample - b0_sample.repeat(x_levels).reshape(len(b0_sample), x_levels))
b_sample = convert_deflection(a0_sample, a_sample, x_levels, y)

#b0_sample = b0_sample * y_sd + y_mean
#b_sample = b_sample * y_sd

#sig_sample = sigma_sample * y_sd
#b_sd_sample = a_sd_sample * y_sd
sig_sample = convert_sigma(y, sigma_sample)
b_sd_sample = convert_sigma(y, a_sd_sample)

# Plot the results.

plot.figure(figsize=(6.0, 4.0))

plot.subplot(211)
plot_post(sig_sample, title=r'$\sigma$ (cell SD) posterior')

plot.subplot(212)
plot_post(b_sd_sample, title=r'$aSD$ posterior')

plot.subplots_adjust(wspace=0.2, hspace=0.5)

plot.figure(figsize=(18.0, 3.0))
total_subplot = len(b_sample[0, :])
plot_n = 100 + (total_subplot + 1) * 10 + 1

plot.subplot(plot_n)
plot_post(b0_sample, title=r'$\beta_0$ posterior')

for i in range(total_subplot):
    plot.subplot(plot_n + i + 1)
    plot_post(b_sample[:, i], title=r'$\beta_{1%i}$ posterior' % (i + 1))

plot.subplots_adjust(wspace=0.2)

n_cons = len(contrasts)
if n_cons > 0:
    plot_per_rows = 5
    plot_rows = ceil(n_cons / plot_per_rows)
    plot_cols = ceil(n_cons / plot_rows)

    plot.figure(figsize=(3.75 * plot_cols, 2.5 * plot_rows))

    for i in range(n_cons):
        contrast = contrasts[i, :]
        comp = np.dot(b_sample, contrast)
        plot.subplot(plot_rows, plot_cols, i + 1)
        plot_post(comp, title='Contrast %i' % (i + 1), comp=0.0)

plot.subplots_adjust(wspace=0.2, hspace=0.5)
plot.show()

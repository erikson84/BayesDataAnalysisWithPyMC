# -*- coding: utf-8 -*-
'''Hierarchical Model for inferring the mean (mu)
and precision (tau) of normal likelihood data via MCMC.
Python (PyMC) adaptation of the R code from "Doing Bayesian Data Analysis",
by John K. Krushcke.
More info: http://doingbayesiandataanalysis.blogspot.com.br/

'''
from __future__ import division

import pymc
import numpy as np
from matplotlib import pyplot as plot
from plot_post import plot_post

# For simplicity's sake, I will generate random data just like
# the R code in the book.

t_mean = 100
t_sd = 15
N = 20

# Generate N samples, no rounding.

y = np.random.normal(t_mean, t_sd, N)

# Defining the priors for mu and tau.

mu = pymc.Normal('mu', 0.0, 1.0e-10)  # Mean: 0.0, SD: 100000
tau = pymc.Gamma('tau', 0.01, 0.01)  # Mean: 1.0, SD: 10

# Now the likelihood function.

like = pymc.Normal('like', mu, tau, value=y, observed=True)

# Create the model, generate initialization values and sample its posterior.

model = pymc.Model([like, mu, tau])
map_ = pymc.MAP(model)
map_.fit()
mcmc = pymc.MCMC(model)
mcmc.sample(iter=60000, burn=40000, thin=2)

# Sample the posterior for the parameter estimates.

mu_sample = mcmc.trace('mu')[:]
tau_sample = mcmc.trace('tau')[:]

# Keeping the same idea as the book: convert the posterior samples to SD.

sigma_sample = 1 / np.sqrt(tau_sample)

# Plot the results.

plot.figure(figsize=(8.0, 8.0))

plot.subplot(211)
plot_post(mu_sample, title=r'$\mu$ posterior distribution')

plot.subplot(212)
plot_post(sigma_sample, title=r'$\sigma$ posterior distribution')

plot.subplots_adjust(wspace=0.2, hspace=0.2)
plot.show()

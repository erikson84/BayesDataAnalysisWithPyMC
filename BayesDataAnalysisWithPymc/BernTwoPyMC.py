# -*- coding: utf-8 -*-
''' Model for inferring two binomial proportions via MCMC.
Python (PyMC) adaptation of the R code from "Doing Bayesian Data Analysis",
by John K. Krushcke.
More info: http://doingbayesiandataanalysis.blogspot.com.br/

'''
from __future__ import division

import pymc
from matplotlib import pyplot as plot
from plot_post import plot_post

# TODO: It would be good to import data from CSV files.

# Model specification in PyMC goes backwards, in comparison to JAGS:
# first the prior are specified, THEN the likelihood function.

# TODO: With PyMC, it´s possible to define many stochastic variables
# in just one variable name using the 'size' function parameter.

# But for now, I will use multiple variable names for simplicity.

theta1 = pymc.Beta('theta1', alpha=3, beta=3)
theta2 = pymc.Beta('theta2', alpha=3, beta=3)

# Define the observed data.

data = [[1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]

# Define the likelihood function for the observed data.

like1 = pymc.Bernoulli('like1', theta1, observed=True, value=data[0])
like2 = pymc.Bernoulli('like2', theta2, observed=True, value=data[1])

# Use the PyMC 'Model' class to collect all the variables we are interested in.

model = pymc.Model([theta1, theta2])

# And instantiate the MCMC class to sample the posterior.

mcmc = pymc.MCMC(model)
mcmc.sample(40000, 10000, 1)

# Use PyMC built-in plot function to show graphs of the samples.

# pymc.Matplot.plot(mcmc)
# plot.show()

# Let's try plotting using Matplotlib's 'pyplot'.
# First, we extract the traces for the parameters of interest.

theta1_samples = mcmc.trace('theta1')[:]
theta2_samples = mcmc.trace('theta2')[:]
theta_diff = theta2_samples - theta1_samples

# Then, we plot a histogram of their individual sample values.

plot.figure(figsize=(8.0, 10))

plot.subplot(311)
plot_post(theta1_samples, title=r'Posterior of $\theta_1$')

plot.subplot(312)
plot_post(theta2_samples, title=r'Posterior of $\theta_2$')

plot.subplot(313)
plot_post(theta_diff, title=r'Posterior of $\Delta\theta$', comp=0.0)

plot.subplots_adjust(hspace=0.5)
plot.show()

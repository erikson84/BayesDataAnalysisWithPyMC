# -*- coding: utf-8 -*-
'''Hierarchical Model for inferring the mean (mu)
and sample size (kappa) of various Bernoulli trials via MCMC.
Python (PyMC) adaptation of the R code from "Doing Bayesian Data Analysis",
by John K. Krushcke.
More info: http://doingbayesiandataanalysis.blogspot.com.br/

'''
from __future__ import division

import pymc
from matplotlib import pyplot as plot
from plot_post import plot_post

# For better code flow, we define the data first.
# Based on the original code's 'Therapeutic touch data'.

z = [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
     4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 8]
N = 10  # Number of trials for each z.

data = [[0] * (N - i) + [1] * i for i in z]  # Build the Bernoulli trial data.

# Again, with PyMC we design the model from top to bottom.
# Let's start defining the hierarchical prior's constants.
# A,B constant for the overall beta distribution for mu.

a_mu = 2.0
b_mu = 2.0

# Shape and rate constants for the overall gamma distribution for kappa.
# They are reparametrized as mean and standard deviation.

s_kappa = 10**2 / 10**2
r_kappa = 10 / 10**2

# Then, we define the overall beta and gamma distributions.

mu = pymc.Beta('mu', a_mu, b_mu)
kappa = pymc.Gamma('kappa', s_kappa, r_kappa)

# Instead of using a 'for' loop for multiple stochastic variables,
# we use the 'size' parameter of PyMC. This is why we defined the data first.
# We could use a '@deterministic' wrapper, but operations already generate it.

a = mu * kappa
b = (1.0 - mu) * kappa

theta = pymc.Beta('theta', a, b, size=len(data))  # One beta for each subject.

# The priors are defined. Now we need to set the likelihood of our data.
# The likelihood can't be defined the same way. We need a 'for' loop.
# Or the 'Lambda()' class.
# For more info: https://github.com/pymc-devs/pymc/issues/319
#
# for i in range(len(data)):
#     like_i = pymc.Bernoulli('like_%i' % i, p=theta[i], value=data[i],
#                             observed=True)
#
# The code above works nicely (the posterior result is the same, and each theta
# is updated with its data. But how does looping the declaration of the same
# variable works? I prefer the following code, since it makes more sense.

like = []
for i in range(len(data)):
    like.append(pymc.Bernoulli('like_%i' % i, p=theta[i],
                               value=data[i], observed=True))

# Done! Now we need to collect the variables and fit our model.

model = pymc.Model([theta, mu, kappa])

map_ = pymc.MAP(model)
map_.fit()

mcmc = pymc.MCMC(model)
mcmc.sample(iter=60000, burn=10000, thin=2)

# Extracting the parameter samples.

mu_sample = mcmc.trace('mu')[:]
kappa_sample = mcmc.trace('kappa')[:]
theta_sample = mcmc.trace('theta')[:]

# And plot them.

plot.figure(figsize=(8.0, 8.0))

plot.subplot(221)
plot_post(mu_sample, comp=0.5, title=r'$\mu$ posterior distribution')

plot.subplot(222)
plot_post(kappa_sample, title=r'$\kappa$ posterior distribution')

plot.subplot(223)
plot_post(theta_sample[:, 0], title=r'$\theta_1$ posterior distribution')

plot.subplot(224)
plot_post(theta_sample[:, 27], title=r'$\theta_{28}$ posterior distribution')

plot.subplots_adjust(wspace=0.2, hspace=0.2)
plot.show()

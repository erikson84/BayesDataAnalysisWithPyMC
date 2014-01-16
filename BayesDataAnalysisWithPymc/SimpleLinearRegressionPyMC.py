# -*- coding: utf-8 -*-
'''Hierarchical Model for estimation of simple linear regression
parameter via MCMC.
Python (PyMC) adaptation of the R code from "Doing Bayesian Data Analysis",
by John K. Krushcke.
More info: http://doingbayesiandataanalysis.blogspot.com.br/

'''
from __future__ import division

import pymc
import numpy as np
from matplotlib import pyplot as plot
from plot_post import plot_post
from normalize import (normalize, convert_intercept,
                       convert_slope, convert_tau_sigma)
from os import path

# Code to find the data path.

scr_dir = path.dirname(__file__)
file_name = 'McIntyre1994data.csv'
comp_dir = path.join(scr_dir, 'Data', file_name)

# So, let's be lazy: the data are from McIntyre cigarette weight.
# Use numpy to load the data we want directly in the appropriate variables.

y, x = np.genfromtxt(comp_dir, delimiter=',',
                     skip_header=1, usecols=(1, 3), unpack=True)

# Let's try normalizing, as suggested by Krushcke.

zy = normalize(y)
zx = normalize(x)

# Define the priors for the model.
# First, normal priors for the slope and intercept.

beta0 = pymc.Normal('b0', 0.0, 1.0e-10)
beta1 = pymc.Normal('b1', 0.0, 1.0e-10)

# Then, gamma and uniform prior for precision and DoF.
# Krushcke suggests the use of a Student's t distribution for the likelihood.
# It makes the estimation more robust in the presence of outliers.
# We will use Krushcke's DoF transformation using a gain constant.

tau = pymc.Gamma('tau', 0.01, 0.01)
udf = pymc.Uniform('udf', 0.0, 1.0)
tdf_gain = 1


@pymc.deterministic
def tdf(udf=udf, tdf_gain=tdf_gain):
    return 1 - tdf_gain * np.log(1 - udf)

# Defining the linear relationship between variables.


@pymc.deterministic
def mu(beta0=beta0, beta1=beta1, x=zx):
    mu = beta0 + beta1 * x
    return mu


# Finally, the likelihood using Student's t distribution.

like = pymc.NoncentralT('like', mu=mu, lam=tau, nu=tdf,
                        value=zy, observed=True)

# For those who want a more traditional linear model:
#like = pymc.Normal('like', mu=mu, tau=tau, value=zy, observed=True)

# The model is ready! Sampling code below.

model = pymc.Model([beta0, beta1, tau, tdf])
fit = pymc.MAP(model)
fit.fit()
mcmc = pymc.MCMC(model)
mcmc.sample(iter=100000, burn=50000, thin=10)

# Collect the sample values for the parameters.

z0_sample = mcmc.trace('b0')[:]
z1_sample = mcmc.trace('b1')[:]
ztau_sample = mcmc.trace('tau')[:]
tdf_sample = mcmc.trace('tdf')[:]

# Convert the data back to scale.

b0_sample = convert_intercept(x, y, z0_sample, z1_sample)
b1_sample = convert_slope(x, y, z1_sample)
sigma_sample = convert_tau_sigma(y, ztau_sample)

# Plot the results

plot.figure(figsize=(8.0, 8.0))

plot.subplot(221)
plot_post(b0_sample, title=r'$\beta_0$ posterior')

plot.subplot(222)
plot_post(b1_sample, title=r'$\beta_1$ posterior')

plot.subplot(223)
plot_post(sigma_sample, title=r'$\sigma$ posterior')

plot.subplot(224)
plot_post(tdf_sample, title=r'tDF posterior')

plot.subplots_adjust(wspace=0.2, hspace=0.2)

# Plot the data with some credible regression lines.

plot.figure(figsize=(8.0, 8.0))

plot.scatter(x, y, c='k', s=60)
plot.title('Data points with credible regression lines')

x1 = plot.axis()[0]
x2 = plot.axis()[1]

plot.autoscale(enable=False)

for line in range(0, len(b1_sample), len(b1_sample) // 50):
    plot.plot([x1, x2], [b0_sample[line] + b1_sample[line] * x1,
                         b0_sample[line] + b1_sample[line] * x2],
              c='#348ABD', lw=1)

plot.show()

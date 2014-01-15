# -*- coding: utf-8 -*-
'''Function to normalize data and convert parameter back to original scale.
Python (PyMC) adaptation of the R code from "Doing Bayesian Data Analysis",
by John K. Krushcke.
More info: http://doingbayesiandataanalysis.blogspot.com.br/

'''
from __future__ import division
import numpy as np


def normalize(data):
    '''Normalizes a set of data.

    '''

    mean = np.mean(data)
    sd = np.sqrt(np.var(data))
    z_data = (data - mean) / sd
    return z_data


def convert_slope(x_data, y_data, zb1_sample):
    '''Converts normalized b1 sample back to original scale.

    :Arguments:
    x_data: original predictor data list.
    y_data: original predicted data list.
    zb1_sample: normalized parameter samples.

    '''

    y_sd = np.sqrt(np.var(y_data))
    x_sd = np.sqrt(np.var(x_data))
    b1 = zb1_sample * (y_sd / x_sd)
    return b1


def convert_intercept(x_data, y_data, zb0_sample, zb1_sample):
    '''Converts normalized b0 sample back to original scale.

    :Arguments:
    x_data: original predictor data list.
    y_data: original predicted data list.
    zb0_sample: normalized parameter samples.
    zb1_sample: normalized parameter samples.

    '''

    y_sd = np.sqrt(np.var(y_data))
    y_mean = np.mean(y_data)

    x_sd = np.sqrt(np.var(x_data))
    x_mean = np.mean(x_data)

    b0 = zb0_sample * y_sd + y_mean - zb1_sample * (y_sd * x_mean) / x_sd
    return b0


def convert_tau_sigma(y_data, ztau_sample):
    '''Converts normalized tau samples back to original scale SD.

    :Arguments:
    y_data: original predicted data list.
    ztau_sample: normalized tau parameter samples.

    '''
    z_sigma = 1 / np.sqrt(ztau_sample)
    y_sd = np.sqrt(np.var(y_data))
    sigma = z_sigma * y_sd
    return sigma

def convert_sigma(y_data, zsigma_sample):
    '''Converts normalized tau samples back to original scale SD.

    :Arguments:
    y_data: original predicted data list.
    ztau_sample: normalized sigma parameter samples.

    '''
    y_sd = np.sqrt(np.var(y_data))
    sigma = zsigma_sample * y_sd
    return sigma


def convert_baseline(a0_sample, a_sample, x_levels, y_data):
    '''Convert normalized ANOVA baseline back to original scale.

    :Arguments:
    a0_sample: normalized baseline samples.
    a_sample: normalized deflection samples.
    x_levels: integer, levels of categorical variable.
    y_data: original predicted data list.

    '''
    m_sample = a0_sample.repeat(x_levels).reshape(len(a0_sample), x_levels) \
    + a_sample
    b0_sample = m_sample.mean(axis=1)
    b0_sample = b0_sample * np.sqrt(np.var(y_data) + np.mean(y_data))
    return b0_sample


def convert_deflection(a0_sample, a_sample, x_levels, y_data):
    '''Convert normalized ANOVA deflections back to original scale.

    :Arguments:
    a0_sample: normalized baseline samples.
    a_sample: normalized deflection samples.
    x_levels: integer, levels of categorical variable.
    y_data: original predicted data list.

    '''
    m_sample = a0_sample.repeat(x_levels).reshape(len(a0_sample), x_levels) \
    + a_sample
    b0_sample = m_sample.mean(axis=1)
    b_sample = (m_sample -
                b0_sample.repeat(x_levels).reshape(len(b0_sample), x_levels))
    b_sample = b_sample * np.sqrt(np.var(y_data))
    return b_sample

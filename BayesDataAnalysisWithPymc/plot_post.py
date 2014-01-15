# -*- coding: utf-8 -*-
'''Plot the histogram of the posterior distribution sample,
with the mean and the 95% HDI.
Adaptation of the R code from "Doing Bayesian Data Analysis",
by John K. Krushcke.
More info: http://doingbayesiandataanalysis.blogspot.com.br/

Histogram code based on (copied from!) 'Probabilistic Programming and
Bayesian Methods for Hackers', by Cameron Davidson-Pilon.
More info: https://github.com/CamDavidsonPilon/
Probabilistic-Programming-and-Bayesian-Methods-for-Hackers

'''

from __future__ import division

from short_hdi import short_hdi
from matplotlib import pyplot as plot


def plot_post(sample, title='Posterior',
              cred=0.95, comp=None, *args, **kwargs):
    '''Plot the histogram of the posterior distribution sample,
    with the mean and the HDI.

    :Arguments:
        sample: array of sample values.
        cred: credible interval (default: 95%)
        comp: value for comparison (default: None)
        title: String value for graph title.

    '''
    # First we compute the shortest HDI using Krushcke's algorithm.

    sample_hdi = short_hdi(sample)

    # Then we plot the histogram of the sample.
    ax = plot.hist(sample,
                   bins=25,
                   alpha=0.85,
                   label='',
                   normed=True)

    # Force the y-axis to be limited to 1.1 times the max probability density.
    maxy = 1.1 * max(ax[0])
    plot.ylim(0.0, maxy)

    # No y-axis label, they are not important here.
    plot.yticks([])

    # Should we plot a vertical line on the mean?
    #plot.vlines(sample.mean(), 0, maxy, linestyle='--',
    #       label=r'Mean (%0.3f)' % sample.mean())
    # But we keep the mean value in its right place.

    plot.text(sample.mean(), 0.9 * max(ax[0]), 'Mean: %0.3f' % sample.mean())

    #plot.legend(loc='upper right') #Legends are cumbersome!
    plot.title(title)

    # Plot the HDI as a vertical line with their respective values.
    plot.hlines(y=0, xmin=sample_hdi[0], xmax=sample_hdi[1], linewidth=6)
    plot.text(sample_hdi[0], max(ax[0]) / 20, '%0.3f' % sample_hdi[0],
          horizontalalignment='center')
    plot.text(sample_hdi[1], max(ax[0]) / 20, '%0.3f' % sample_hdi[1],
          horizontalalignment='center')

    # In case there is a comparison value, plot it and
    # compute how much of the posterior falls at each side.
    if comp != None:
        loc = max(ax[0]) / 2.0
        plot.vlines(comp, 0, loc, color='green', linestyle='--')
        less = 100 * (sum(sample < comp)) / len(sample)
        more = 100 * (sum(sample > comp)) / len(sample)
        print less, more
        plot.text(comp, loc, '%0.1f%% < %0.1f < %0.1f%%' % (less, comp, more),
                  color='green', horizontalalignment='center')

    #return ax  # I thought the function should return something. It's not needed.

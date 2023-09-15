import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import tensorflow as tf
import sys
from shap_utils import *
from Shapley import ShapNN
from scipy.stats import spearmanr
import shutil
from sklearn.base import clone
import matplotlib.pyplot as plt
import warnings
import itertools
import inspect
import _pickle as pkl
from sklearn.metrics import f1_score, roc_auc_score

def performance_plots(self, vals, name=None,
                      num_plot_markers=20, sources=None):
    """Plots the effect of removing valuable points.

    Args:
        vals: A list of different valuations of data points each
             in the format of an array in the same length of the data.
        name: Name of the saved plot if not None.
        num_plot_markers: number of points in each plot.
        sources: If values are for sources of data points rather than
               individual points. In the format of an assignment array
               or dict.

    Returns:
        Plots showing the change in performance as points are removed
        from most valuable to least.
    """
    plt.rcParams['figure.figsize'] = 8, 8
    plt.rcParams['font.size'] = 25
    plt.xlabel('Fraction of train data removed (%)')
    plt.ylabel('Prediction accuracy (%)', fontsize=20)
    if not isinstance(vals, list) and not isinstance(vals, tuple):
        vals = [vals]
    if sources is None:
        sources = {i: np.array([i]) for i in range(len(self.X))}
    elif not isinstance(sources, dict):
        sources = {i: np.where(sources == i)[0] for i in set(sources)}
    vals_sources = [np.array([np.sum(val[sources[i]])
                              for i in range(len(sources.keys()))])
                    for val in vals]
    if len(sources.keys()) < num_plot_markers:
        num_plot_markers = len(sources.keys()) - 1
    plot_points = np.arange(
        0,
        max(len(sources.keys()) - 10, num_plot_markers),
        max(len(sources.keys()) // num_plot_markers, 1)
    )
    perfs = [self._portion_performance(
        np.argsort(vals_source)[::-1], plot_points, sources=sources)
        for vals_source in vals_sources]
    rnd = np.mean([self._portion_performance(
        np.random.permutation(np.argsort(vals_sources[0])[::-1]),
        plot_points, sources=sources) for _ in range(10)], 0)
    plt.plot(plot_points / len(self.X) * 100, perfs[0] * 100,
             '-', lw=5, ms=10, color='b')
    if len(vals) == 3:
        plt.plot(plot_points / len(self.X) * 100, perfs[1] * 100,
                 '--', lw=5, ms=10, color='orange')
        legends = ['TMC-Shapley ', 'G-Shapley ', 'LOO', 'Random']
    elif len(vals) == 2:
        legends = ['TMC-Shapley ', 'LOO', 'Random']
    else:
        legends = ['TMC-Shapley ', 'Random']
    plt.plot(plot_points / len(self.X) * 100, perfs[-1] * 100,
             '-.', lw=5, ms=10, color='g')
    plt.plot(plot_points / len(self.X) * 100, rnd * 100,
             ':', lw=5, ms=10, color='r')
    plt.legend(legends)
    if self.directory is not None and name is not None:
        plt.savefig(os.path.join(
            self.directory, 'plots', '{}.png'.format(name)),
            bbox_inches='tight')
        plt.close()
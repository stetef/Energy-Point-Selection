"""Module contains commonly used functions analysis."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import *

import umap

mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = ['Arial']

def read_tddft_spectrum_file(path):
    """Read spectrum file."""
    return np.loadtxt(path).T

def format_axis(ax, energyrange=None, ticks=(5, 10), fontsize=20):
    """Format axes for spectra plots."""
    ax.set_yticks([])
    ax.set_xlim(energyrange)
    ax.set_xlabel('Energy (eV)', fontsize=fontsize)
    ax.xaxis.set_minor_locator(MultipleLocator(ticks[0]))
    ax.xaxis.set_major_locator(MultipleLocator(ticks[1]))
    ax.set_xticklabels(np.array(ax.get_xticks(), dtype=int), fontsize=fontsize - 2)
    ax.tick_params(direction='in', width=3, length=13, which='major')
    ax.tick_params(direction='in', width=2, length=7, which='minor')

def parse_file(filename):
    """Parse datafile from experiment."""
    data = []
    with open(filename) as f:
        for line in f:
            parsed = line.replace('\t', ' ').replace('\n','').split(' ')
            while "" in parsed:
                parsed.remove('')
            data.append(parsed)
    f.close()
    columns = [ele.replace('_', ' ') for ele in data[0]]
    df = pd.DataFrame(data[1:], columns=columns)
    return df

def get_coeffs(n, dropout):
    """Randomly generate coeffs that add to one."""
    while True:
        coeffs = np.random.rand(n)
        proba = np.random.rand(n)
        set_to_zero = proba < dropout
        coeffs[set_to_zero] = 0
        if sum(coeffs) != 0:
            break
    scale = 1/np.sum(coeffs)
    coeffs = coeffs * scale
    return coeffs

def generate_linear_combos(Refs, scale=0, N=10, dropout=0.5, training=True):
    """Create linear combo dataset from Refs."""
    n = len(Refs)
    Data = []
    Coeffs = []
    for i in range(N):
        coeffs = get_coeffs(n, dropout)
        if scale != 0:
            noise = np.random.normal(scale=scale,
                                     size=Refs.shape[1])
        else:
            noise = 0
        Data.append(Refs.T @ coeffs + noise)
        Coeffs.append(coeffs)
    if training:
        return np.array(Data), -np.ones((N, 1))
    else:
        return np.array(Data), np.array(Coeffs)

def make_scree_plot(data, n=5, threshold=0.95, show_first_PC=True):
    """
    Make PCA scree plot.
    
    Attributes:
        data - Data on which to perform PCA analysis.
            type = ndarray
    kwargs:
        n - The number of PC's to display/ keep on the x-axis.
            default = 5
            type = int
        threshold - The variance cutoff to display.
            default = 0.95
            type = float
        show_first_PC - Opt to display the first PC.
            default = True
            type = boolean
    """
    fig, ax = plt.subplots(figsize=(8,6))
    pca = PCA()
    pca_components = pca.fit_transform(data)

    x = np.arange(n) + 1
    cdf = [np.sum(pca.explained_variance_ratio_[:i + 1]) for i in range(n)]

    ax.plot(x, cdf, 's-', markersize=10, fillstyle='none',
            color=plt.cm.tab10(.15))
    ax.plot(x, np.ones(len(x)) * threshold, 'k--', linewidth=3)

    if show_first_PC:
        PC1 = pca.components_[0]
        plt.plot(np.linspace(1, 5, len(PC1)), -PC1*0.3 + min(cdf) + 0.1, 'k', linewidth=2)
        text = ax.text(1, min(cdf) + 0.13, '$PC_1$', ha="left", va="center", size=20)

    plt.xticks(x, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(min(cdf) - 0.05, 1.02)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Number of Parameters', fontsize=22)
    plt.ylabel(f'Cumultative\nExplained Variance', fontsize=22)
    ax.tick_params(direction='in', width=2, length=8)
    plt.show()

def visualize_energy_points(plot, Energy, Refs, energy_points,
                            fontsize=20, ticks=(5, 10), label=None):
    """Visualize results from RFE on reference spectra."""
    fig, ax = plot
    for i, ref in enumerate(Refs):
        ax.plot(Energy, ref, '-', linewidth=2, c=plt.cm.tab20(i))
    format_axis(ax, ticks=ticks, fontsize=fontsize-2)
    for energy in energy_points:
        ax.axvline(energy, c='gray', linestyle='--', linewidth=0.5)
    if label is not None:
        ax.set_title(label, fontsize=fontsize)

def scale_coeffs_to_add_to_one(coeff_mtx):
    """For every coeff list in coeff_mtx, scale to add to one."""
    return np.array([coeffs/np.sum(coeffs) for coeffs in coeff_mtx])

def mae_loss(coeffs, Refs, target, metric='mean_absolute_error'):
    calc = Refs.T @ coeffs
    calc = calc - np.min(calc)  # set min to zero
    return eval(metric)(calc, target) + 10*(np.sum(coeffs) - 1)**2

def get_coeffs_from_sepctra(spectra, Refs):
    """Find the coeffs that minimize spectral recontruction error."""
    m = Refs.shape[0]
    coeffs_0 = np.ones(m)/m
    bounds = np.zeros((m, 2))
    bounds[:, 1] = 1
    coeffs = np.array([minimize(mae_loss, coeffs_0,
                       args=(Refs, spectrum), bounds=bounds)['x']
                       for spectrum in spectra])
    return scale_coeffs_to_add_to_one(coeffs)

def plot_reconstructions(data, coeffs, m, Energy, Refs, metric='mean_absolute_error',
                         verbose=True):
    """Recon plot of spectra in a row."""
    fig, axes = plt.subplots(figsize=(5*m, 5), ncols=m)
    plt.subplots_adjust(wspace=0)
    for i in range(m):
        pred = Refs.T @ coeffs[i]
        pred = pred - np.min(pred)
        true = data[i]
        ax = axes[i]
        ax.plot(Energy, pred, '-', linewidth=4, c=plt.cm.tab20(0), label='predicted')
        ax.plot(Energy, true, '-', linewidth=4, c=plt.cm.tab20(2), label='true')
        format_axis(ax, ticks=(10, 20), fontsize=20)
        ax.legend(fontsize=20, loc=4)
    plt.show()
    metric_name = metric.replace('_', ' ')
    if verbose:
        print(f'{metric_name}: {eval(metric)(pred, true)}')
        return eval(metric)(pred, true)

def plot_spectra(Energy, data, mod=0, alpha=0.01):
    """Spaghetti plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, spectrum in enumerate(data):
        if mod != 0:
            if i%mod == 0:
                ax.plot(Energy, spectrum , '-', linewidth=2, c='gray', alpha=alpha)
        else:
            ax.plot(Energy, spectrum , '-', linewidth=2, c='gray', alpha=alpha)
    format_axis(ax)
    plt.show()

def two_dimensional_visualization(data, Refs, data_columns, N_refs=None, ncol=2,
                                  method='PCA', fontsize=18):
    """Plot dataset and Refs in 2D using dimensionality reduction."""
    fig, ax = plt.subplots(figsize=(4, 4))

    if method == 'PCA':
        reducer = PCA(random_state=42, n_components=2)
        reducer = reducer.fit(data)
    elif method == 'UMAP':
        reducer = umap.UMAP(random_state=42, n_components=2)
        reducer = reducer.fit(data)
    elif method == 't-SNE':
        tsne = TSNE(random_state=42, n_components=2)
        total_data = np.array([x for y in [data, Refs] for x in y])
        transformed_data = tsne.fit_transform(total_data)

    if method in ['PCA', 'UMAP']:
        transformed = reducer.transform(data)
        transformed_Refs = reducer.transform(Refs)
    else:
        transformed = transformed_data[:-len(Refs)]
        transformed_Refs = transformed_data[-len(Refs):]

    for i, coords in enumerate(transformed):
        ax.plot(coords[0], coords[1], 'o', markersize=10, c='gray', alpha=0.05)

    n = Refs.shape[0]
    counts = 0
    ele_to_idx_map = {'Cd': 2, 'Te': 8, 'Se': 6, 'Cl': 4, 'exp': 1, 'As': 0}

    for i, coords in enumerate(transformed_Refs):
        if N_refs is None:
            label = data_columns[i]
        else:
            label = data_columns[N_refs[i]]
        for key in ele_to_idx_map.keys():
            if key in label:
                j = ele_to_idx_map[key]
                break
        if n < 20:
            c = counts
        else:
            c = j
        color = plt.cm.tab20(c)
        counts += 1
        ax.plot(coords[0], coords[1], 'o', markersize=10, c=color, label=label)

    ax.plot(np.average(transformed_Refs[:,0]),
            np.average(transformed_Refs[:,1]),
            'o', markersize=10, c='k', label='centroid')

    ax.legend(fontsize=fontsize - 4, loc='center left', bbox_to_anchor=(1., .5), ncol=ncol)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(f'${method}_1$', fontsize=fontsize)
    ax.set_ylabel(f'${method}_2$', fontsize=fontsize)
    plt.show()

def set_spine_width(ax, width=2):
    for spine in ['top','bottom','left','right']:
        ax.spines[spine].set_linewidth(width)

def histogram(plot, x, bins=50, color=plt.cm.tab20(2), fontsize=18, ticks=(1, 10),
              label_map=None):
    fig, ax = plot
    n, bin_vals, patches = plt.hist(x, bins=bins, range=(0, bins),
                                    color=color, edgecolor='w')
    plt.xlim(-1, bins)
    ax.set_ylabel('Counts', fontsize=fontsize + 2)
    ax.set_xlabel('Index', fontsize=fontsize + 2)
    ax.xaxis.set_minor_locator(MultipleLocator(ticks[0]))
    ax.xaxis.set_major_locator(MultipleLocator(ticks[1]))
    ax.set_yticklabels(np.array(ax.get_yticks(), dtype=int), fontsize=fontsize)
    ax.set_xticklabels(np.array(ax.get_xticks(), dtype=int), fontsize=fontsize)
    ax.tick_params(direction='out', width=3, length=13, which='major', axis='x')
    ax.tick_params(direction='out', width=2, length=7, which='minor', axis='x')
    ax.tick_params(direction='in', width=3, length=13, which='major', axis='y')
    if label_map is not None:
        for idx, label in label_map.items():
            rect = patches[idx]
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label,
                    ha='center', va='bottom', fontsize=fontsize - 2)

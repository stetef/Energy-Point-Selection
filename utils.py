"""Module contains commonly used functions analysis."""

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = ['Arial']

def read_tddft_spectrum_file(path):
    """Read spectrum file."""
    return np.loadtxt(path).T

def format_axis(ax, energyrange=None, ticks=(5, 10), fontsize=20):
    """Format axes for spectral viewing."""
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

def get_coeffs(n):
    """Randomly generate coeffs that add to one."""
    coeffs = np.random.rand(n)
    scale = 1/np.sum(coeffs)
    coeffs = coeffs * scale
    return coeffs

def generate_linear_combos(Refs, scale=0, N=10):
    """Create linear combo dataset."""
    n = len(Refs)
    Data = []
    Coeffs = []
    for i in range(N):
        coeffs = get_coeffs(n)
        if scale != 0:
            noise = np.random.normal(scale=scale,
                                     size=Refs.shape[1])
        else:
            noise = 0
        Data.append(Refs.T @ coeffs + noise)
        Coeffs.append(coeffs)
    return np.array(Data), np.array(Coeffs)

def visualize_energy_points(plot, Energy, Refs, energy_points,
                            fontsize=20, ticks=(5, 10), label=None):
    fig, ax = plot
    for i, ref in enumerate(Refs):
        ax.plot(Energy, ref, '-', linewidth=2, c=plt.cm.tab20(i))
    format_axis(ax, ticks=ticks, fontsize=fontsize-2)
    for energy in energy_points:
        ax.axvline(energy, c='gray', linestyle='--', linewidth=0.5)
    if label is not None:
        ax.set_title(label, fontsize=fontsize)

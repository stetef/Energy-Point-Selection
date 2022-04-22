"""Module contains commonly used functions analysis."""
import os, re

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *

import umap

mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = ['Arial']

def read_tddft_spectrum_file(path):
    """Read spectrum file."""
    return np.loadtxt(path).T

def format_axis(ax, energyrange=None, ticks=(5, 10), fontsize=20, xlabel='Energy (eV)'):
    """Format axes for spectra plots."""
    ax.set_yticks([])
    ax.set_xlim(energyrange)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.xaxis.set_minor_locator(MultipleLocator(ticks[0]))
    ax.xaxis.set_major_locator(MultipleLocator(ticks[1]))
    ax.set_xticks(ax.get_xticks())
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
        if np.sum(coeffs) != 0:
            break
    scale = 1 / np.sum(coeffs)
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
        x = Refs.T @ coeffs
        x = x - np.min(x)
        Data.append(x + noise)
        Coeffs.append(coeffs)
    Data = np.array(Data)
    #Data = Data - np.min(Data)
    if training:
        return Data, -np.ones((N, 1))
    else:
        return Data, np.array(Coeffs)

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
    return np.array([coeffs / np.sum(coeffs) for coeffs in coeff_mtx])

def objective_function(coeffs, Refs, target, metric, lambda1, lambda2):
    calc = Refs.T @ coeffs
    calc = calc - np.min(calc)  # set min to zero
    return eval(metric)(calc, target) \
           + lambda1 * np.sum(np.abs(coeffs)) \
           + lambda2 * (np.sum(coeffs) - 1)**2

def objective_function_with_scaling(coeffs, Refs, target, metric, lambda1, lambda2):
    calc = Refs.T @ coeffs[1:]
    calc = calc - np.min(calc)  # set min to zero
    calc = calc * coeffs[0]  # first coeff := scaling factor
    return eval(metric)(calc, target) \
           + lambda1 * np.sum(np.abs(coeffs[1:])) \
           + lambda2 * (np.sum(coeffs[1:]) - 1)**2

def filter_coeffs_from_tol(coefficients, tol):
    filtered_coeffs = []
    for coeffs in coefficients:
        bool_arr = coeffs < tol
        if np.sum(bool_arr) == len(coeffs):
            print('tolerance too big. choosing max concentration.')
            idx = np.argmax(coeffs)
            coeffs = np.zeros(len(coeffs))
            coeffs[idx] = 1
        else:
            coeffs[bool_arr] = 0
            coeffs = scale_coeffs_to_add_to_one([coeffs])[0]
        filtered_coeffs.append(coeffs)
    return np.array(filtered_coeffs)

def get_coeffs_from_spectra(spectra, Refs, scaling=False, tol=None,
                            metric='mean_squared_error',
                            lambda1=0.2, lambda2=2.):
    """Find the coeffs that minimize spectral recontruction error."""
    m = Refs.shape[0]
    if scaling:
        coeffs_0 = np.ones(m + 1) / (m + 1)
        bounds = np.zeros((m + 1, 2))
        bounds[:, 1] = 1
        bounds[0, 1] = 20
        coeffs = np.array([minimize(objective_function_with_scaling, coeffs_0,
                           args=(Refs, spectrum, metric, lambda1, lambda2),
                           bounds=bounds)['x']
                           for spectrum in spectra])
        scale, coefficients = coeffs[:, 0], scale_coeffs_to_add_to_one(coeffs[:, 1:])
        if tol is not None:
            coefficients = filter_coeffs_from_tol(coefficients, tol)
        return scale, coefficients
    else:
        coeffs_0 = np.ones(m)/m
        bounds = np.zeros((m, 2))
        bounds[:, 1] = 1
        coeffs = np.array([minimize(objective_function, coeffs_0,
                           args=(Refs, spectrum, metric, lambda1, lambda2),
                           bounds=bounds)['x']
                           for spectrum in spectra])
        coefficients = scale_coeffs_to_add_to_one(coeffs)
        if tol is not None:
            coefficients = filter_coeffs_from_tol(coefficients, tol)
        return coefficients

def plot_reconstructions(plot, data, coeffs, m, Energy, Refs, verbose=True, leg=True,
                         metric='mean_absolute_error', scale=None, color=2):
    """Recon plot of spectra in a row."""
    fig, axes = plot
    preds, truths = [], []
    for i in range(m):
        pred = Refs.T @ coeffs[i]
        if scale is not None:
            pred = pred * scale[i]
        pred = pred - np.min(pred)
        preds.append(pred)
        true = data[i]
        true = true - np.min(true)
        truths.append(truths)
        ax = axes[i]
        ax.plot(Energy, pred, '-', linewidth=4, c=plt.cm.tab20(color), label='predicted')
        ax.plot(Energy, true, '-', linewidth=4, c=plt.cm.tab20(14), label='true')
        format_axis(ax, ticks=(10, 20), fontsize=20)
        if leg:
            ax.legend(fontsize=20, loc=4)
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

def get_metrics(indices, data, coeffs, test_data, test_coeffs, Refs,
                metric='explained_variance_score', verbose=True):
    metrics = []
    for i in range(len(indices)):
        idx = indices[:i + 1]
        score = train_RF(Refs, idx, data, coeffs, test_data, test_coeff,
                         metric=metric) 
        if verbose:
            print(f' n = {i}: score = {score}')
        metrics.append(score)
    return np.array(metrics)

def train_RF(Refs, idx, train_data, train_coeffs, test_data, test_coeff,
             metric="explained_variance_score", spectra=True):
    
    reg = RandomForestRegressor(n_estimators=30)
    reg.fit(train_data[:, idx], train_coeffs)

    pred_coeff = reg.predict(test_data[:, idx])

    test_scores = []
    m = len(test_data)
    for j in range(m):
        if spectra:
            pred_spectra = Refs.T @ pred_coeff[j]
            pred_spectra = pred_spectra - np.min(pred_spectra)
            true_spectra = test_data[j]
            temp_score = eval(metric)(pred_spectra, true_spectra)
        else:
            temp_score = eval(metric)(pred_coeff, test_coeff)
        test_scores.append(temp_score)
    score = np.average(test_scores)
    return score

def preprocess_df(df):
    headers = df.iloc[0]
    df = df.rename(columns=headers)
    df = df.drop(0)
    return df

def get_data_from_set(Set, n_pts, Energies, path='Experimental Data/'):
    """Returns list of dictionaries."""
    
    directory_regex = re.compile(f'.*Set {Set}.*')
    file_regex = re.compile('.*.csv')
    
    file_names = []
    for root, dirs, files in os.walk(path):
        for directory in dirs:
            if directory_regex.match(directory):
                files = os.listdir(path + directory)
                for file in files:
                    if file_regex.match(file):
                        file_names.append(path + directory + '/' + file)
    
    exp_data_dfs = []
    for filename in file_names:
        exp_data_dfs.append(pd.read_csv(filename))
    
    Exp_Data = [{} for i in range(n_pts)]
    for i in range(9):
        df = exp_data_dfs[i]
        pixels = np.array(df['pixel_number'], dtype=int)
        
        if Set in [1, 2, 4, 5]:
            x_pos = np.array(df[' axis_pos[mm]'], dtype=float)
            y_pos = np.array(df[' yaxis_pos[mm]'], dtype=float)
        elif Set == 3:
            x_pos = np.array(df[' xaxis_pos[mm]'], dtype=float)
            y_pos = np.array(df[' axis_pos[mm]'], dtype=float) 
        
        I = df[' As[]']
        E = Energies[i]

        for j in range(n_pts):
            pixel = pixels[j]
            entry = Exp_Data[j]
            if entry == {}:
                Exp_Data[j] = {'pixel': pixel, 'x': x_pos[j], 'y': y_pos[j], 'E': [E],
                               'I': [I[j]]}
            else:
                Exp_Data[j]['I'].append(I[j])
                Exp_Data[j]['E'].append(E)
    
    return Exp_Data

def get_all_data(Energies, unit='eV'):
    Alldata = {}
    if unit == 'eV':
        Energies = Energies* 1000  
    for Set in range(1, 6):
        if Set in [1, 2, 4, 5]:
            n_pts = 15
        elif Set == 3:
            n_pts = 5
        Alldata[Set] =  get_data_from_set(Set, n_pts, Energies)
    return Alldata

def set_spine_width(ax, width=2):
    for spine in ['top','bottom','left','right']:
        ax.spines[spine].set_linewidth(width)

def turn_off_spines(ax, spines=['top','bottom','left','right']):
    for spine in spines:
        ax.spines[spine].set_visible(False)

def get_xy_from_colms(df, clm1, clm2):
    """Get arrays from dataframe df."""
    x = np.array(df[clm1], dtype=float)
    x = x * 1000  # convert to eV
    y = np.array(df[clm2], dtype=float)
    y = y - np.min(y)
    y = y / np.max(y)
    return x, y

def get_label_and_color(column_name):
    ele_to_idx_map = {'Cd': 2, 'Te': 8, 'Se': 6, 'Cl': 4, 'exp': 1, 'As': 0}

    for key in ele_to_idx_map.keys():
        if key in column_name:
            c = ele_to_idx_map[key]
            if key == 'exp':
                label = column_name.replace('exp ', '*').replace('standard', 'ref.')
            else:
                label = column_name.replace('merged', '').replace('symmetrized', '').replace('mp ', '')
            break
    return label, c

def make_conc_bar_chart(plot, coeffs, data_columns, width=0.75, offset=0,
                        varcolor=0, format_ticks=True):
    
    m = coeffs.shape[0]
    fig, ax = plot
    labels = ["$" + "P_{" + f"{i}" + "}$" for i in range(1, m + 1)]
    colors = [plt.cm.tab20(varcolor), plt.cm.tab20(14)]

    for i in range(coeffs.shape[0]):
        conc_map = {num: coeffs[i, num] for num in range(coeffs.shape[1])}
        sorted_conc_map = {k: v*100 for k, v in sorted(conc_map.items(),
                           key=lambda item: item[1], reverse=True)}
        bottoms = [np.sum(list(sorted_conc_map.values())[:tmp], axis=0)
                   for tmp in range(coeffs.shape[1])]
        keys = list(sorted_conc_map.keys())
        for k, val in enumerate(list(sorted_conc_map.values())):
            if val != 0:
                key = keys[k]
                xlabel = labels[i]
                #color = colors[k%2]
                color = colors[0]
                bottom = bottoms[k]
                rect = ax.bar(i + offset, val, width, label=k, bottom=bottom,
                              fc=color, edgecolor='k', linewidth=2.5)
                ax.bar_label(rect, labels=[key + 1], label_type='center', c='w',
                             fontsize=18)
            
    set_spine_width(ax, width=2)

    ax.tick_params(direction='out', width=2, length=10, which='major', axis='both')
    ax.set_ylabel('Concentration (%)', fontsize=20)
    if format_ticks:
        ax.set_xticks(ax.get_xticks()[1:-1])
        ax.set_xticklabels(labels, fontsize=20)
        ax.set_yticks(np.array(ax.get_yticks())[:-1])
        ax.set_yticklabels(np.array(ax.get_yticks(), dtype=int), fontsize=18)

def get_error(coeffs, target, Refs, scale,
              metric='mean_absolute_error'):
    pred_spectrum = Refs.T @ coeffs
    pred_spectrum = pred_spectrum * scale
    pred_spectrum = pred_spectrum - np.min(pred_spectrum)
    targte = target - np.min(target)
    return eval(metric)(pred_spectrum, target)

def get_errors_with_different_noises(Y_Refs, noises=np.arange(0, 0.06, 0.01),
                                     metric='mean_squared_error', lambda1=0.1):
    """Return spectral recon error given minimization process for scaled vs unscaled spectra."""
    Errors = []
    for scale in noises:
        print(f'Noise: {scale * 100}%')

        N = 50
        kwargs = {'N': N, 'scale': scale, 'dropout': 0.9, 'training': False}
        test_data, test_coeffs = generate_linear_combos(Y_Refs, **kwargs)

        unscaled_coeffs = get_coeffs_from_spectra(test_data, Y_Refs, scaling=False,
                                                  lambda1=lambda1, metric=metric)
        scales, scaled_coeffs = get_coeffs_from_spectra(test_data, Y_Refs, scaling=True,
                                                        lambda1=lambda1, metric=metric)

        errors = [[], [], []]
        i = 0
        for coeffs, scale in zip([test_coeffs, unscaled_coeffs, scaled_coeffs],
                                 [np.ones(N),  np.ones(N),      scales]):
            for j in range(N):
                error = get_error(coeffs[j], test_data[j], Y_Refs, scale[j], metric=metric)
                errors[i].append(error)
            i += 1
        Errors.append(errors)
    return np.array(Errors)

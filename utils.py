"""Module contains commonly used functions analysis."""
import os, re

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
from sklearn.metrics.pairwise import cosine_similarity

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
    columns = [ele.replace('  ', '_') for ele in columns]
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

def get_uncert_from_min_fun(result, ftol=2.220446049250313e-09):
    # ftol obtained from
    # https://docs.scipy.org/doc/scipy/reference/
    # optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
    I = np.identity(len(result.x))
    sigmas = np.zeros(len(result.x))
    for i in range(len(result.x)):
        hess_inv_i = result.hess_inv(I[i, :])[i]  # get diagonal entry
        uncertainty_i = np.sqrt(max(1, abs(result.fun)) * ftol * hess_inv_i)
        sigmas[i] = uncertainty_i
    return sigmas

def scale_coeffs_to_add_to_one(coeff_mtx, sigmas=None):
    """For every coeff list in coeff_mtx, scale to add to one."""
    sums = [np.sum(coeffs) for coeffs in coeff_mtx]
    normalized_coeffs = np.array([coeff_mtx[i] / sums[i] for i in range(len(sums))])
    if sigmas is not None:
        rescaled_sigmas = np.array([sigmas[i] / sums[i] for i in range(len(sums))])
        return normalized_coeffs, rescaled_sigmas
    else:
        return normalized_coeffs, None

def objective_function(coeffs, Refs, target, lambda1, lambda2):
    calc = Refs.T @ coeffs
    #calc = calc - np.min(calc)  # set min to zero
    return np.sum((calc - target)**2) \
           + lambda1 * np.sum(np.abs(coeffs)) \
           + lambda2 * (np.sum(coeffs) - 1)**2

def objective_function_with_scaling(coeffs, Refs, target, lambda1, lambda2):
    calc = Refs.T @ coeffs[1:]
    #calc = calc - np.min(calc)  # set min to zero
    calc = calc * coeffs[0]  # first coeff := scaling factor
    return np.sum((calc - target)**2) \
           + lambda1 * np.sum(np.abs(coeffs[1:])) \
           + lambda2 * (np.sum(coeffs[1:]) - 1)**2

def filter_coeffs_from_tol(coefficients, tol, sigmas=None):
    filtered_coeffs = []
    filtered_sigmas = []
    for i, coeffs in enumerate(coefficients):
        if sigmas is not None:
            sigmas_i = sigmas[i]
        bool_arr = coeffs < tol
        if np.sum(bool_arr) == len(coeffs):
            print('tolerance too big. choosing max concentration.')
            idx = np.argmax(coeffs)
            coeffs = np.zeros(len(coeffs))
            coeffs[idx] = 1
            if sigmas is not None:
                sigmas_i = np.zeros(len(coeffs))
                sigmas_i[idx] = sigmas[i, idx]
        else:
            coeffs[bool_arr] = 0
            if sigmas is not None:
                sigmas_i[bool_arr] = 0
                coeffs, sigmas_i = scale_coeffs_to_add_to_one([coeffs], sigmas=[sigmas_i])
                coeffs = coeffs[0]
                sigmas_i = sigmas_i[0]
            else:
                coeffs, _ = scale_coeffs_to_add_to_one([coeffs])
                coeffs = coeffs[0]
        filtered_coeffs.append(coeffs)
        if sigmas is not None:
            filtered_sigmas.append(sigmas_i)
    return np.array(filtered_coeffs), np.array(filtered_sigmas)

def get_coeffs_from_spectra(spectra, Refs, scaling=False, tol=None,
                            lambda1=10, lambda2=1e8, return_sigmas=True):
    """Find the coeffs that minimize spectral recontruction error."""
    m = Refs.shape[0]
    if scaling:
        coeffs_0 = np.ones(m + 1) / (m + 1)
        bounds = np.zeros((m + 1, 2))
        bounds[:, 1] = 1
        bounds[0, 1] = 20
        results = [minimize(objective_function_with_scaling, coeffs_0,
                   args=(Refs, spectrum, lambda1, lambda2), bounds=bounds)
                   for spectrum in spectra]
        coeffs = np.array([results[i].x for i in range(len(results))])
        if return_sigmas:
            sigmas = np.array([get_uncert_from_min_fun(results[i]) for i in range(len(results))])
            scales =  coeffs[:, 0]
            coefficients, rescaled_sigmas = scale_coeffs_to_add_to_one(coeffs[:, 1:], sigmas=sigmas[:, 1:])
            if tol is not None:
                coefficients, rescaled_sigmas = filter_coeffs_from_tol(coefficients, tol, sigmas=rescaled_sigmas)
            return scales, coefficients, rescaled_sigmas
        else:
            scales =  coeffs[:, 0]
            coefficients, _ = scale_coeffs_to_add_to_one(coeffs[:, 1:])
            if tol is not None:
                coefficients, _ = filter_coeffs_from_tol(coefficients, tol)
            return scales, coefficients, None
    else:
        coeffs_0 = np.ones(m)/m
        bounds = np.zeros((m, 2))
        bounds[:, 1] = 1
        results = [minimize(objective_function, coeffs_0,
                   args=(Refs, spectrum, lambda1, lambda2),
                   bounds=bounds) for spectrum in spectra]
        coeffs = np.array([results[i].x for i in range(len(results))])
        sigmas = np.array([get_uncert_from_min_fun(results[i]) for i in range(len(results))])
        coefficients, rescaled_sigmas = scale_coeffs_to_add_to_one(coeffs, sigmas)
        if tol is not None:
            coefficients, rescaled_sigmas = filter_coeffs_from_tol(coefficients, tol, rescaled_sigmas)
        return coefficients, rescaled_sigmas

def plot_reconstructions(plot, data, coeffs, m, Energy, Refs, verbose=True,
                         metric='mean_squared_error', scale=None, color=2,
                         fontsize=18):
    """Recon plot of spectra in a row."""
    fig, axes = plot
    preds, truths = [], []
    for i in range(m):
        pred = coeffs[i] @ Refs
        pred = pred - np.min(pred)
        if scale is not None:
            pred = pred * scale[i]
        preds.append(pred)
        true = data[i]
        truths.append(truths)
        ax = axes[i]
        if i == m -1:
            leg = True
        else:
            leg = False
        ax.plot(Energy, pred, '-', linewidth=4, c=plt.cm.tab20(color), label='predicted')
        ax.plot(Energy, true, '-', linewidth=4, c=plt.cm.tab20(14), label='true')
        if verbose:
            format_axis(ax, ticks=(10, 20), fontsize=16)
            ax.set_xlim(11862, 11918)
        if leg:
            ax.legend(fontsize=fontsize, loc='center left', bbox_to_anchor=(1, 0.5))
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

def make_conc_bar_chart(plot, coeffs, data_columns, yerrs, width=0.75, offset=0,
                        varcolor=0, format_ticks=True):
    
    m = coeffs.shape[0]
    fig, ax = plot
    labels = ["$" + "P_{" + f"{i}" + "}$" for i in range(1, m + 1)]
    colors = [plt.cm.tab20(varcolor), plt.cm.tab20(14)]

    for i in range(m):
        conc_map = {num: coeffs[i, num] for num in range(coeffs.shape[1])}
        sorted_conc_map = {idx: conc*100 for idx, conc in sorted(conc_map.items(),
                           key=lambda item: item[1], reverse=True)}
        bottoms = [np.sum(list(sorted_conc_map.values())[:tmp], axis=0)
                   for tmp in range(coeffs.shape[1])]
        keys = list(sorted_conc_map.keys())

        if yerrs is not None:
            yerrs_sorted = {num: yerrs[i, num] * 100
                            for num, conc in sorted(conc_map.items(),
                            key=lambda item: item[1], reverse=True)}

        for k, conc in enumerate(list(sorted_conc_map.values())):
            if conc != 0:
                key = keys[k]
                xlabel = labels[i]
                color = colors[0]
                bottom = bottoms[k]
                error_kw = {'linewidth': 1.5, 'ecolor': 'w', 'capsize': 4}
                if yerrs is None:
                    yerr = None
                else:
                    yerr = yerrs_sorted[k]
                rect = ax.bar(i + offset, conc, width, yerr=yerr, 
                              label=k, bottom=bottom, error_kw=error_kw,
                              fc=color, edgecolor='k', linewidth=1.)
                ax.bar_label(rect, labels=[key + 1], label_type='center', c='w',
                             fontsize=18, fontweight='bold')
            
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

def evaluate_similarity(x, y, metric):
    if metric == 'cosine similarity':
        score = cosine_similarity([x], Y=[y])[0][0]
    elif metric == 'Pearson correlation':
        score, pval = pearsonr(x, y)
    elif metric == '1 - $\delta$':
        score = 1 - np.average(np.abs(x - y))
    elif metric == '1 - MSE':
        metric = 'mean_squared_error'
        score = 1 - eval(metric)(x, y)
    elif metric == '1 - IADR':
        score = 1 - np.sum(np.abs(x - y)) / max(np.sum(x), np.sum(y))
    return score

def get_sets_from_subset_indices(subset_indices, basis):
    subset = np.array([ele for i, ele in enumerate(basis) if i in subset_indices])
    non_subset_indices = np.array([i for i, ele in enumerate(basis) if i not in subset_indices])
    non_subset = np.array([ele for i, ele in enumerate(basis) if i not in subset_indices])
    return subset, non_subset_indices, non_subset

def plot_hist_scores_per_element(plot, hist, data_columns, subset_size):
    fig, ax = plot
    n = len(hist)
    for i in range(n):
        if hist[i] != []:
            ax.bar(i, np.average(hist[i]), color=plt.cm.tab20(0), edgecolor='w')
    
    ax.tick_params(direction='out', width=2, length=6, axis='both')
    plt.setp(ax.get_yticklabels(), fontsize=14)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(data_columns, fontsize=12, rotation=90)
    ylabel = r"$\widebar{R}_{N = " + f"{subset_size}" + r"}$"
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlim(-1, n)

def get_score_from_subset(subset, non_subset, metric):
    _, coeffs_hat, _ = get_coeffs_from_spectra(non_subset, subset, scaling=True, tol=None,
                                               lambda1=0, lambda2=2.)
    recons = coeffs_hat @ subset
    scores = [evaluate_similarity(x, y, metric) for x, y in zip(non_subset, recons)]
    return np.average(scores)

def initialize_random_select_from_cuts(segments, sample_sizes, n):
    cut_indices = get_cuts(np.arange(n), segments, sample_sizes)
    subset_indices = []
    for sample_size, cut_idxs in zip(sample_sizes, cut_indices):
        cut_selection = np.random.choice(cut_idxs, size=sample_size, replace=False)
        for i in cut_selection:
            subset_indices.append(i)
    return np.sort(np.array(subset_indices))

def get_cut_num_from_i(idx, cut_to_idx_map):
    for cut_num, cut_idxs in cut_to_idx_map.items():
        if idx in cut_idxs:
            return cut_num

def get_test_i_from_cut_num(non_subset_indices, cut_to_replace, cut_to_idx_map):
    non_subset_indices_in_cut = [idx for idx in list(cut_to_idx_map[cut_to_replace]) if idx in non_subset_indices]
    return np.random.choice(non_subset_indices_in_cut, size=1, replace=False)[0]

def sample_test_i_from_cuts(subset_indices, non_subset_indices, cut_sizes, cut_to_idx_map):
    n = np.sum(cut_sizes)
    
    replace_subset_i = np.random.randint(0, high=len(subset_indices))
    replace_i = subset_indices[replace_subset_i]
    cut_to_replace = get_cut_num_from_i(replace_i, cut_to_idx_map)
    
    test_i = get_test_i_from_cut_num(non_subset_indices, cut_to_replace, cut_to_idx_map)
    
    return replace_subset_i, test_i

def get_cuts(data, segments, sample_sizes):
    cuts = []
    for i in range(len(segments) - 1):
        k = segments[i]
        l = segments[i + 1]
        cut = [data[j] for j in range(k, l)]
        cuts.append(cut)
    return cuts

def plot_error_vs_lambda(plot, Unscaled_errors, Scaled_errors,
                         num_contribs_unscale, num_contribs_scale, p=0.5):
    lambdas = np.array(list(Unscaled_errors.keys()))
    
    unscal = np.array(list(Unscaled_errors.values())) * 10
    unscal_contribs = np.array(list(num_contribs_unscale.values()))

    scal = np.array(list(Scaled_errors.values())) * 10
    scal_contribs = np.array(list(num_contribs_scale.values()))
    
    fig, axes = plot
    
    y1 = [unscal, unscal_contribs, p*unscal + (1 - p)*unscal_contribs]
    y2 = [scal, scal_contribs, p*scal + (1 - p)*scal_contribs]
    ylabels = ['$MSE$', '$\overline{N}_{contributions}$', '$' + f'{p:.2f}' +
               '*MSE + ' + f'{1-p:.2f}' + '*\overline{N}_{contributions}$']

    for i, ax in enumerate(axes):
        ax.plot(lambdas, y1[i], 'x-', c=plt.cm.tab20(0), label='unscaled')
        ax.plot(lambdas, y2[i], 'o-', c=plt.cm.tab20(2), label='scaled')

        if i == 1:
            ax.legend(fontsize=18)
        ax.tick_params(direction='out', width=2, length=8, which='major', axis='both')
        #if i == 1:
        #    ax.yaxis.set_major_locator(MultipleLocator(2.))
        ax.set_ylabel(ylabels[i], fontsize=18)
        ax.set_xlabel('$\lambda_1$', fontsize=18)
        plt.setp(ax.get_xticklabels(), fontsize=16)
        plt.setp(ax.get_yticklabels(), fontsize=16)
        set_spine_width(ax, width=1.5)

def monte_carlo_sample(basis, subset, subset_indices, non_subset, non_subset_indices, subset_size,
                       metric, beta, cut_sizes, cut_to_idx_map):
    
    if subset_size is not None:
        replace_subset_i = np.random.randint(0, high=len(subset_indices))
        test_basis_i = np.random.randint(0, high=len(non_subset_indices))
    else:
        replace_subset_i, test_basis_i = sample_test_i_from_cuts(subset_indices, non_subset_indices,
                                                             cut_sizes, cut_to_idx_map)
    
    test_subset_indices = subset_indices.copy()
    test_subset_indices[replace_subset_i] = test_basis_i
    
    test_subset, test_non_subset_indices, test_non_subset = get_sets_from_subset_indices(test_subset_indices, basis)
    
    avg_score = get_score_from_subset(subset, non_subset, metric)
    test_avg_score = get_score_from_subset(test_subset, test_non_subset, metric)
    
    delta_u = avg_score - test_avg_score
    if test_avg_score < avg_score:
        print(f'P={np.exp(-beta * delta_u)}')
    if test_avg_score > avg_score:
        print('Replacing')
        return test_subset, test_subset_indices, test_non_subset_indices, test_non_subset, test_avg_score
    elif test_avg_score < avg_score and np.random.random() < np.exp(-beta * delta_u):
        print(f'Replacing cuz T')
        return test_subset, test_subset_indices, test_non_subset_indices, test_non_subset, test_avg_score
    else:
        print("Keeping")
        return subset, subset_indices, non_subset_indices, non_subset, avg_score

def sample_basis(basis, num_iters, subset_size, segments=None, sample_sizes=None,
                 cut_sizes=None, cut_to_idx_map=None, metric='cosine similarity',
                 monte_carlo=False, beta=3e5, step_size=1e4):
    indices = np.arange(basis.shape[0])
    total_score = []
    histogram_per_basis_element = [[] for i in range(basis.shape[0])]
    
    for i in range(num_iters):
        print(i + 1, end='\r')
                
        if monte_carlo:
            if i == 0:
                if subset_size is None:
                    subset_indices = initialize_random_select_from_cuts(segments, sample_sizes, len(basis))
                else:
                    subset_indices = np.random.choice(indices, size=subset_size, replace=False)
                subset, non_subset_indices, non_subset = get_sets_from_subset_indices(subset_indices, basis)    
            else:
                subset, subset_indices, non_subset_indices, non_subset, score = monte_carlo_sample(
                    basis, subset, subset_indices, non_subset, non_subset_indices, subset_size, 
                    metric, beta, cut_sizes, cut_to_idx_map)
                beta =+ step_size  
        else:
            subset_indices = np.random.choice(indices, size=subset_size, replace=False)
            subset, non_subset_indices, non_subset = get_sets_from_subset_indices(subset_indices, basis)
        
        score = get_score_from_subset(subset, non_subset, metric)
        total_score.append(score)
        for idx in subset_indices:
            histogram_per_basis_element[idx].append(score)
    
    if monte_carlo:
        print(subset_indices)
        return np.array(total_score), subset_indices
    else:
        return np.array(total_score), histogram_per_basis_element

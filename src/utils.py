"""
utils.py
Subcellular location prediction utilities
Author: Ramon Viñas, 2018
Contact: ramon.torne.17@ucl.ac.uk
"""
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import pandas as pd

# AMINOACID UTILS
AA_NAMES = {'A': 'alanine',
            'B': 'aspartate/asparagine',
            'C': 'cystine',
            'D': 'aspartate',
            'E': 'glutamate',
            'F': 'phenylalanine',
            'G': 'glycine',
            'H': 'histidine',
            'I': 'isoleucine',
            'K': 'lysine',
            'L': 'leucine',
            'M': 'methionine',
            'N': 'asparagine',
            'P': 'proline',
            'Q': 'glutamine',
            'R': 'arginine',
            'S': 'serine',
            'T': 'threonine',
            'U': 'selenocysteine',
            'V': 'valine',
            'W': 'tryptophan',
            'Y': 'tyrosine',
            'Z': 'glutamate/glutamine'}
AA_NAMES_LIST = list(AA_NAMES.keys())

AA_CODES = {'alanine': {'A': 1},
            'cystine': {'C': 1},
            'aspartate': {'D': 1},
            'glutamate': {'E': 1},
            'phenylalanine': {'F': 1},
            'glycine': {'G': 1},
            'histidine': {'H': 1},
            'isoleucine': {'I': 1},
            'lysine': {'K': 1},
            'leucine': {'L': 1},
            'methionine': {'M': 1},
            'asparagine': {'N': 1},
            'proline': {'P': 1},
            'glutamine': {'Q': 1},
            'arginine': {'R': 1},
            'serine': {'S': 1},
            'threonine': {'T': 1},
            'valine': {'V': 1},
            'tryptophan': {'W': 1},
            'tyrosine': {'Y': 1}}
AA_CODES_LIST = list([list(aa.keys())[0] for aa in AA_CODES.values()])
AA_CODES.update({'selenocysteine': {'U': 1}})  # 21st aminoacid

AA_SPECIAL = {'X': 'any',
              '*': 'translation stop',
              '-': 'gap of indeterminate length'}
AA_SPECIAL_LIST = list(AA_SPECIAL.keys())


def aa_wild_prior(code, aa, aa_list):
    # TODO: Do we have a better prior knowledge?
    return 1 / len(aa_list)


AA_WILD = {'U': ['C'],
           'X': AA_CODES_LIST,
           'B': ['D', 'N'],
           'Z': ['E', 'Q']}
AA_WILD_LIST = list(AA_WILD.keys())
for code, v in AA_WILD.items():
    for aa in v:
        AA_CODES[AA_NAMES[aa]][code] = aa_wild_prior(code, aa, v)

AA_MOLECULAR_WEIGHTS = {'A': 89.1,  # g/mol
                        'C': 121.2,
                        'D': 133.1,
                        'E': 147.1,
                        'F': 165.2,
                        'G': 75.1,
                        'H': 155.2,
                        'I': 131.2,
                        'K': 146.2,
                        'L': 131.2,
                        'M': 149.2,
                        'N': 132.1,
                        'P': 115.1,
                        'Q': 146.2,
                        'R': 174.2,
                        'S': 105.1,
                        'T': 119.1,
                        'V': 117.1,
                        'W': 204.2,
                        'Y': 181.2}
AA_MOLECULAR_WEIGHTS_MEAN = np.mean(list(AA_MOLECULAR_WEIGHTS.values()))
AA_MOLECULAR_WEIGHTS_STD = np.std(list(AA_MOLECULAR_WEIGHTS.values()))
AA_MOLECULAR_WEIGHTS.update({'U': 167.1,  # 21st aminoacid
                             'B': aa_wild_prior('B', 'D', ['D', 'N']) * 133.1 + aa_wild_prior('B', 'N',
                                                                                              ['D', 'N']) * 132.1,
                             'Z': aa_wild_prior('Z', 'E', ['E', 'Q']) * 147.1 + aa_wild_prior('Z', 'Q',
                                                                                              ['E', 'Q']) * 146.2,
                             'X': AA_MOLECULAR_WEIGHTS_MEAN})
AA_MOLECULAR_WEIGHTS_NORM = {k: (v - AA_MOLECULAR_WEIGHTS_MEAN) / AA_MOLECULAR_WEIGHTS_STD for k, v in
                             AA_MOLECULAR_WEIGHTS.items()}

# Kyte & Doolittle index of hydrophobicity
AA_HYDROPHOBICITY = {'A': 1.8,
                     'C': 2.5,
                     'D': -3.5,
                     'E': -3.5,
                     'F': 2.8,
                     'G': -0.4,
                     'H': -3.2,
                     'I': 4.5,
                     'K': -3.9,
                     'L': 3.8,
                     'M': 1.9,
                     'N': -3.5,
                     'P': -1.6,
                     'Q': -3.5,
                     'R': -4.5,
                     'S': -0.8,
                     'T': -0.7,
                     'V': 4.2,
                     'W': -0.9,
                     'Y': -1.3}
AA_HYDROPHOBICITY_MEAN = np.mean(list(AA_HYDROPHOBICITY.values()))
AA_HYDROPHOBICITY_STD = np.std(list(AA_HYDROPHOBICITY.values()))
AA_HYDROPHOBICITY.update({'U': AA_HYDROPHOBICITY['C'],
                          'B': aa_wild_prior('B', 'D', ['D', 'N']) * -3.5 + aa_wild_prior('B', 'N', ['D', 'N']) * -3.5,
                          'Z': aa_wild_prior('Z', 'E', ['E', 'Q']) * -3.5 + aa_wild_prior('Z', 'Q', ['E', 'Q']) * -3.5,
                          'X': AA_HYDROPHOBICITY_MEAN})
AA_HYDROPHOBICITY_NORM = {k: (v - AA_HYDROPHOBICITY_MEAN) / AA_HYDROPHOBICITY_STD for k, v in AA_HYDROPHOBICITY.items()}

# Hopp & Wood index of hydrophilicity
AA_HYDROPHILICITY = {'A': -0.5,
                     'C': -1.0,
                     'D': 3.0,
                     'E': 3.0,
                     'F': -2.5,
                     'G': 0.0,
                     'H': -0.5,
                     'I': -1.8,
                     'K': 3.0,
                     'L': -1.8,
                     'M': -1.3,
                     'N': 0.2,
                     'P': 0.0,
                     'Q': 0.2,
                     'R': 3.0,
                     'S': 0.3,
                     'T': -0.4,
                     'V': -1.5,
                     'W': -3.4,
                     'Y': -2.3}
AA_HYDROPHILICITY_MEAN = np.mean(list(AA_HYDROPHILICITY.values()))
AA_HYDROPHILICITY_STD = np.std(list(AA_HYDROPHILICITY.values()))
AA_HYDROPHILICITY.update({'U': AA_HYDROPHILICITY['C'],
                          'B': aa_wild_prior('B', 'D', ['D', 'N']) * 3.0 + aa_wild_prior('B', 'N', ['D', 'N']) * 0.2,
                          'Z': aa_wild_prior('Z', 'E', ['E', 'Q']) * 3.0 + aa_wild_prior('Z', 'Q', ['E', 'Q']) * 0.2,
                          'X': AA_HYDROPHILICITY_MEAN})
AA_HYDROPHILICITY_NORM = {k: (v - AA_HYDROPHILICITY_MEAN) / AA_HYDROPHILICITY_STD for k, v in AA_HYDROPHILICITY.items()}
AA_CHARGE = {'K': 10.0, 'R': 12.0, 'H': 5.98}


# AMINOACID UTILITY FUNCTIONS
def replace_wild_first(seq):
    for aa, v in AA_WILD.items():
        seq = seq.replace(aa, v[0])
    return seq


def replace_selenocysteine(seq):
    return seq.replace('U', 'C')


def aa_composition(seq):
    # TODO: Do not distribute wildcards?
    # Count aminoacids and compute sequence lengths
    c = Counter(seq)
    aa_counts = {aa: c[aa] for aa in AA_CODES_LIST}
    aa_wild_counts = {aa: c[aa] for aa in AA_WILD_LIST}
    seq_len = sum(list(aa_counts.values()) + list(aa_wild_counts.values()))
    # ASSERTED ASSUMPTION: no '*' and '-' appear in the current dataset
    for k, wild_c in aa_wild_counts.items():
        wild_aas = AA_WILD[k]
        for waa in wild_aas:
            aa_counts[waa] += wild_c * aa_wild_prior(k, waa, wild_aas)
    rel_counts = [aa_counts[aa] / seq_len for aa in AA_CODES_LIST]
    # rel_counts holds the relative counts aligned according AA_CODES_LIST
    # rel_counts*seq_len are the absolute counts

    # Compute molecular weight
    mol_weight = sum([v * AA_MOLECULAR_WEIGHTS[k] for k, v in aa_counts.items()])
    return rel_counts, seq_len, mol_weight, AA_CODES_LIST


def _compute_psaac_tau(seq, k):
    L = len(seq)
    J = 0
    for aa_1, aa_2 in zip(seq, seq[k:]):
        J += ((AA_HYDROPHILICITY_NORM[aa_1] - AA_HYDROPHILICITY_NORM[aa_2]) ** 2 +
              (AA_HYDROPHOBICITY_NORM[aa_1] - AA_HYDROPHOBICITY_NORM[aa_2]) ** 2 +
              (AA_MOLECULAR_WEIGHTS_NORM[aa_1] - AA_MOLECULAR_WEIGHTS_NORM[aa_2]) ** 2) / 3
    return J / (L - k)


def pseudo_aa_composition(seq, lambd, w=0.05):
    f, seq_len, mol_weight, aa_list = aa_composition(seq)
    taus = np.array([_compute_psaac_tau(seq, k) for k in range(2, lambd + 1)])
    norm = (np.sum(f) + w * np.sum(taus))
    f = f / norm
    p = (w * taus) / norm
    return f, p, seq_len, mol_weight, aa_list


def seq_hydrophobicity(seq):
    return np.mean([AA_HYDROPHOBICITY[aa] for aa in seq])


def seq_hydrophilicity(seq):
    return np.mean([AA_HYDROPHILICITY[aa] for aa in seq])


def seq_molecular_weight(seq):
    return np.mean([AA_MOLECULAR_WEIGHTS[aa] for aa in seq])


# FLEXIBILITY SCORE
def flexibility_index(scores):
    # Accuracy of protein flexibility predictions.
    # Scores are the flexibility scores obtained by sliding a window of length 9, as described in
    # "Accuracy of protein flexibility predictions". Vihinen M. et al. 1994
    n = len(scores)
    return np.sum(scores[9: n - 9]) / n


# CROSS-VALIDATION UTILS
def get_val_split(y_train):
    y_train_ = y_train
    if y_train.ndim == 2:  # one-hot
        y_train_ = np.argmax(y_train, axis=1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return next(skf.split(np.zeros_like(y_train_), y_train_))


def get_test_split(y):
    y_ = y
    if y.ndim == 2:  # one-hot
        y_ = np.argmax(y, axis=1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    train_idxs, test_idxs = next(skf.split(np.zeros_like(y_), y_))
    return train_idxs, test_idxs


def zip_ngram(seq, n):
    return zip(*[seq[i:] for i in range(n)])


def count_aa_ngram(seq, n=2, symmetric=True):
    n_grams = zip_ngram(seq, n)
    keys = AA_CODES_LIST + AA_WILD_LIST
    keys = list(itertools.product(keys, repeat=n))
    c = Counter(n_grams)
    aa_counts = {aa: c[aa] for aa in keys}
    if symmetric:
        aa_counts_new = {}
        for k in keys:
            aa_counts_new[k] = aa_counts[k] + aa_counts[tuple(reversed(k))]
        aa_counts = aa_counts_new
        rel_counts = [aa_counts[aa] / (2 * (len(seq) - n + 1)) for aa in keys]
    else:
        rel_counts = [aa_counts[aa] / (len(seq) - n + 1) for aa in keys]
    return rel_counts, keys


# PLOTTING UTILS
def plot_distribution(data, key='seq_len'):
    sns.distplot(data[key])


def plot_violin(df, class_dict, key='seq_len', threshold=None):
    df_ = df
    if threshold:
        df_ = df[df[key] < threshold]
    plt.subplots(figsize=(9, 7))
    ax = sns.violinplot(x='class', y=key, data=df_)
    class_ids = ax.get_xticklabels()
    class_dict_inv = {v: k for k, v in class_dict.items()}
    x_ticks = [class_dict_inv[int(class_id.get_text())] for class_id in class_ids]
    ax.set_xticklabels(x_ticks)


def aminoacid_corr_heatmap(df, df_filt, freq_key='counts_global_', vmax=.4, title=None):
    columns = [col for col in df.columns.values if col.startswith(freq_key)]
    corr = df_filt[columns].corr() - df[columns].corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.subplots(figsize=(9, 7))
    ax = sns.heatmap(corr, vmax=vmax, mask=mask, cmap=cmap)
    xticks_aa = [tick.get_text().split('_')[-1] for tick in ax.get_xticklabels()]
    new_xticks = ['{} ({})'.format(AA_NAMES[x_tick], x_tick) for x_tick in xticks_aa]
    ax.set_xticklabels(new_xticks)
    ax.set_yticklabels(new_xticks)
    ax.set_title(title)


def _build_aa_matrix(df, freq_key='counts_pairs_'):
    columns = [col for col in df.columns.values if col.startswith(freq_key)]
    new_cols = list(set([col.split('_')[-1] for col in columns]))
    n_aa = len(new_cols)
    aa_pairs_matrix = np.zeros((n_aa, n_aa))
    for i, aa1 in enumerate(new_cols):
        for j, aa2 in enumerate(new_cols):
            aa_pairs_matrix[i, j] = np.mean(df.loc[:, freq_key + '{}_{}'.format(aa1, aa2)].values)
    return pd.DataFrame(aa_pairs_matrix, columns=new_cols, index=new_cols)


def aminoacid_pairs_heatmap(df, df_filt, freq_key='counts_pairs_', vmax=.05, title=None):
    df_ = _build_aa_matrix(df, freq_key)
    df_filt_ = _build_aa_matrix(df_filt, freq_key)
    aa_pairs = df_filt_ - df_
    # print(aa_pairs.columns)
    mask = np.zeros_like(aa_pairs, dtype=np.bool)
    mask[np.triu_indices_from(mask, 1)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.subplots(figsize=(9, 7))
    ax = sns.heatmap(aa_pairs, vmax=vmax, mask=mask, cmap=cmap)
    xticks_aa = [tick.get_text().split('_')[-1] for tick in ax.get_xticklabels()]
    new_xticks = ['{} ({})'.format(AA_NAMES[x_tick], x_tick) for x_tick in xticks_aa]
    ax.set_xticklabels(new_xticks, rotation='vertical')
    ax.set_yticklabels(new_xticks, rotation='horizontal')
    ax.set_title(title)


def densities_joy_plot(df, class_dict, key='seq_len', threshold=None):
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    if threshold:
        df = df[df[key] < threshold]

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row='class', hue='class', aspect=15, size=1, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, key, clip_on=False, shade=True, alpha=0.8, lw=5, bw=.2)
    g.map(sns.kdeplot, key, clip_on=False, color='w', lw=2, bw=.2)
    g.map(plt.axhline, y=0, lw=4, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight='bold', color=color,
                ha='left', va='center', transform=ax.transAxes)

    g.map(label, key)

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play will with overlap
    g.set_titles('')
    g.set(yticks=[])
    g.despine(bottom=True, left=True)


def plot_confusion_matrix(y_pred, y_test, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = confusion_matrix(y_test, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_roc_curve(y_pred, y_true, class_dict):
    n_classes = len(class_dict)
    if y_true.ndim == 1:
        y_true = np.eye(n_classes)[y_true]

    for k, i in class_dict.items():
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(tpr, fpr, label='{} (AUC={})'.format(k, round(roc_auc, 2)))

    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('False Positive Rate')
    plt.xlabel('Sensitivity')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='upper left')
    plt.show()

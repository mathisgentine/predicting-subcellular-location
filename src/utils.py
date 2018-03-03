"""
utils.py
Utilities
"""
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

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
            'selenocysteine': {'U': 1},  # 21st aminoacid
            'valine': {'V': 1},
            'tryptophan': {'W': 1},
            'tyrosine': {'Y': 1}}
AA_CODES_LIST = list([list(aa.keys())[0] for aa in AA_CODES.values()])

AA_SPECIAL = {'X': 'any',
              '*': 'translation stop',
              '-': 'gap of indeterminate length'}
AA_SPECIAL_LIST = list(AA_SPECIAL.keys())


def aa_wild_prior(code, aa, aa_list):
    # TODO: Do we have a better prior knowledge?
    return 1 / len(aa_list)


AA_MOLECULAR_WEIGHTS = {'A': 89.1,  # g/mol
                        'B': aa_wild_prior('B', 'D', ['D', 'N']) * 133.1 + aa_wild_prior('B', 'N', ['D', 'N']) * 132.1,
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
                        'U': 167.1,  # 21st aminoacid
                        'V': 117.1,
                        'W': 204.2,
                        'Y': 181.2,
                        'Z': aa_wild_prior('Z', 'E', ['E', 'Q']) * 147.1 + aa_wild_prior('Z', 'Q', ['E', 'Q']) * 146.2}

AA_WILD = {'X': AA_CODES_LIST,
           'B': ['D', 'N'],
           'Z': ['E', 'Q']}
AA_WILD_LIST = list(AA_WILD.keys())
for code, v in AA_WILD.items():
    for aa in v:
        AA_CODES[AA_NAMES[aa]][code] = aa_wild_prior(code, aa, v)


def replace_wild_first(seq):
    for aa, v in AA_WILD.items():
        seq = seq.replace(aa, v[0])
    return seq


def replace_selenocysteine(seq):
    return seq.replace('U', 'C')


def count_aa(seq):
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


# FLEXIBILITY SCORE
def flexibility_index(scores):
    # Accuracy of protein flexibility predictions.
    # Scores are the flexibility scores obtained by sliding a window of length 9, as described in
    # "Accuracy of protein flexibility predictions". Vihinen M. et al. 1994
    n = len(scores)
    return np.sum(scores[9: n - 9])/n


# CROSS-VALIDATION UTILS
def get_val_split(y_train):
    y_train_ = y_train
    if y_train.ndim == 2:  # one-hot
        y_train_ = np.argmax(y_train, axis=1)

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    return next(skf.split(np.zeros_like(y_train_), y_train_))


# PLOTTING UTILS
def plot_distribution(data, key='seq_len'):
    sns.distplot(data[key])


def plot_violin(df, class_dict, key='seq_len', threshold=2000):
    plt.subplots(figsize=(9, 7))
    ax = sns.violinplot(x='class', y=key, data=df[df[key] < threshold])
    class_ids = ax.get_xticklabels()
    class_dict_inv = {v: k for k, v in class_dict.items()}
    x_ticks = [class_dict_inv[int(class_id.get_text())] for class_id in class_ids]
    ax.set_xticklabels(x_ticks)


def aminoacid_corr_heatmap(df, freq_key='global_rel_', vmax=.4, title=None):
    columns = [col for col in df.columns.values if col.startswith(freq_key)]
    corr = df[columns].corr()
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

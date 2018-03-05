"""
data_pipeline.py
Parses FASTA data and creates features from the aminoacid sequences
Author: Ramon Viñas, 2018
Contact: ramon.torne.17@ucl.ac.uk

FASTA aminoacid codes:
A  alanine               P  proline
B  aspartate/asparagine  Q  glutamine
C  cystine               R  arginine
D  aspartate             S  serine
E  glutamate             T  threonine
F  phenylalanine         U  selenocysteine
G  glycine               V  valine
H  histidine             W  tryptophan
I  isoleucine            Y  tyrosine
K  lysine                Z  glutamate/glutamine
L  leucine               X  any
M  methionine            *  translation stop
N  asparagine            -  gap of indeterminate length
"""
from utils import *
from collections import Counter
import numpy as np

DATA_DIR = '../data'
CLASSES = ['cyto', 'mito', 'nucleus', 'secreted']
N_CLASSES = len(CLASSES)
TEST = 'blind'


def _parse_fasta(data, lines, c):
    lines_iter = iter(lines)
    line = next(lines_iter, None)
    max_len = 0
    while line:
        assert line.startswith('>')
        info = line
        line = next(lines_iter, None)
        seq = ''
        while line and not line.startswith('>'):
            seq += line.rstrip()
            line = next(lines_iter, None)
        max_len = max((max_len, len(seq)))
        data['info'].append(info)
        data['seq'].append(seq)
        data['class'].append(c)
    return max_len


def _load_data(data, name, c):
    with open('{}/{}.fasta'.format(DATA_DIR, name), 'r') as f:
        lines = f.readlines()
    return _parse_fasta(data, lines, c)


def _get_data():
    train = {'info': [], 'seq': [], 'class': []}
    test = {'info': [], 'seq': [], 'class': []}
    max_len = _load_data(test, TEST, None)
    print('Max sequence lenght test: {}'.format(max_len))
    for c in CLASSES:
        seq_len = _load_data(train, c, c)
        max_len = max((max_len, seq_len))
    print('Max sequence length: {}'.format(max_len))
    return train, test, max_len


def _assertions(data, eps=1e-7):
    # Check that assumptions made on the current dataset are correct
    for seq in data['seq']:
        assert '*' not in seq
        assert '-' not in seq

    # Feature assertions
    if 'counts' in data and 'counts_localfirst' in data and 'counts_locallast' in data:
        for global_rel, local_first_rel, local_last_rel in zip(data['counts'], data['counts_localfirst'],
                                                               data['counts_locallast']):
            assert abs(1 - sum(global_rel)) < eps
            assert abs(1 - sum(local_first_rel)) < eps
            assert abs(1 - sum(local_last_rel)) < eps

    # Useful information
    c = Counter(data['class'])
    print('Class balance: {}'.format(c))


def _create_features(data, local=50, indiv_keys=False):
    # Be careful: some chains have less than 50 aminoacids
    aa_list = None
    seq_lens = []
    counts = []
    counts_localfirst = []
    counts_locallast = []
    mol_weights = []
    hydrophobicity = []
    hydrophilicity = []
    for seq in data['seq']:
        # Count aminoacids, find sequence length and compute molecular weight
        # counts, seq_len, mol_weight, aa_list = aa_composition(seq)
        c, seq_len, mol_weight, aa_list = aa_composition(seq)
        counts.append(c)
        seq_lens.append(seq_len)
        mol_weights.append(mol_weight)

        # Count local_first first aminonacids
        c, _, _, _ = aa_composition(seq[:local])
        counts_localfirst.append(c)

        # Count local_last last aminonacids
        c, _, _, _ = aa_composition(seq[-local:])
        counts_locallast.append(c)

        # Compute hydrophobicity and hydrophilicity
        hydrophobicity.append(seq_hydrophobicity(seq))
        hydrophilicity.append(seq_hydrophilicity(seq))

    data['seq_len'] = np.array(seq_lens)
    data['molecular_weight'] = np.array(mol_weights)
    data['hydrophobicity'] = np.array(hydrophobicity)
    data['hydrophilicity'] = np.array(hydrophilicity)

    counts = np.array(counts)
    counts_localfirst = np.array(counts_localfirst)
    counts_locallast = np.array(counts_locallast)
    if indiv_keys:
        for i, aa in enumerate(aa_list):
            data['counts_{}'.format(aa)] = counts[:, i]
            data['counts_localfirst_{}'.format(aa)] = counts_localfirst[:, i]
            data['counts_locallast_{}'.format(aa)] = counts_locallast[:, i]
    else:
        data['counts'] = counts
        data['counts_localfirst'] = counts_localfirst
        data['counts_locallast'] = counts_locallast
    return aa_list


def _make_numpy(data):
    for k, v in data.items():
        data[k] = np.array(v)


def _create_features_biopython(data, local=50):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    feature_fun = {
        'molecular_weight{}': lambda pa: pa.molecular_weight(),
        'iso_point{}': lambda pa: pa.isoelectric_point(),
        'aromaticity{}': lambda pa: pa.aromaticity(),
        # 'gravy{}': lambda pa: pa.gravy(),
        'instability_index{}': lambda pa: pa.instability_index(),
        'flexibility{}': lambda pa: flexibility_index(pa.flexibility()),
        'secondary_structure_fraction{}': lambda pa: pa.secondary_structure_fraction()
    }

    for k in feature_fun.keys():
        data[k.format('')] = []
        data[k.format('_localfirst')] = []
        data[k.format('_locallast')] = []

    for seq in data['seq']:
        # Global features
        seq = replace_selenocysteine(replace_wild_first(seq))
        pa = ProteinAnalysis(seq)
        for k, fun in feature_fun.items():
            data[k.format('')].append(fun(pa))

        # Local features
        pa = ProteinAnalysis(seq[:local])
        for k, fun in feature_fun.items():
            data[k.format('_localfirst')].append(fun(pa))

        pa = ProteinAnalysis(seq[-local:])
        for k, fun in feature_fun.items():
            data[k.format('_locallast')].append(fun(pa))

    _make_numpy(data)


def _normalize_column(train_arr, test_arr, mode=0):
    concat = np.concatenate((train_arr, test_arr))
    if mode == 0:  # Mean 0, std 1
        mean = np.mean(concat)
        std = np.std(concat)
        train_arr = (train_arr - mean) / std
        test_arr = (test_arr - mean) / std
    elif mode == 1:
        max_val = np.max(concat)
        min_val = np.min(concat)
        train_arr = (train_arr - min_val) / (max_val - min_val)
    return train_arr, test_arr


def _normalize(train, test):
    normalize_features = {'seq_len', 'molecular_weight', 'iso_point',
                          'aromaticity', 'instability_index', 'flexibility',
                          'molecular_weight_localfirst', 'molecular_weight_locallast',
                          'iso_point_localfirst', 'iso_point_locallast',
                          # 'gravy_localfirst', 'gravy_locallast',
                          # 'aromaticity_localfirst', 'aromaticity_locallast',
                          }
    for feature in normalize_features:
        train[feature], test[feature] = _normalize_column(train[feature], test[feature])


def _get_features(data):
    x = np.concatenate((data['seq_len'][:, None],
                        data['counts'],
                        data['counts_localfirst'],
                        data['counts_locallast'],
                        data['molecular_weight'][:, None],
                        data['iso_point'][:, None],
                        data['aromaticity'][:, None],
                        # data['gravy'][:, None],
                        data['instability_index'][:, None],
                        data['secondary_structure_fraction'],
                        data['flexibility'][:, None],
                        data['molecular_weight_localfirst'][:, None],
                        data['molecular_weight_locallast'][:, None],
                        # data['iso_point_localfirst'][:, None],
                        # data['iso_point_locallast'][:, None],
                        # data['local_ssf_first'],
                        # data['local_ssf_last'],
                        # data['gravy_localfirst'][:, None],
                        # data['gravy_locallast'][:, None],
                        # data['aromaticity_localfirst'][:, None],
                        # data['aromaticity_localfirst'][:, None],
                        data['hydrophobicity'][:, None],
                        data['hydrophilicity'][:, None]
                        # data['local_ii_first'][:, None],
                        # data['local_ii_last'][:, None]
                        ), axis=1)
    y = data['class']
    return x, y


def _encode_class(data, one_hot=False):
    class_dict = {c: i for i, c in enumerate(CLASSES)}
    encoded_classes = [class_dict[c] for c in data['class']]
    if one_hot:
        encoded_classes = np.eye(N_CLASSES)[encoded_classes]
    data['class'] = np.array(encoded_classes)
    return class_dict


def _encode_aminoacids(data, aa_dict=None, one_hot=False, pad=None):
    codes = AA_NAMES_LIST + ['X']
    if aa_dict is None:
        aa_dict = {c: i for i, c in enumerate(codes)}
    encoded_sequences = []
    for seq in data['seq']:
        encoded_seq = [aa_dict[aa] for aa in seq]
        seq_len = len(encoded_seq)
        if one_hot:
            encoded_seq = np.eye(len(codes))[encoded_seq]
            if pad is not None:
                encoded_seq = encoded_seq[:pad, :]
                encoded_seq = np.pad(encoded_seq,
                                     ((0, max(pad - seq_len, 0)), (0, 0)),
                                     mode='constant', constant_values=0)
        elif pad is not None:
            encoded_seq = encoded_seq[:pad]
            encoded_seq = np.pad(encoded_seq,
                                 (0, max(pad - seq_len, 0)),
                                 mode='constant', constant_values=0)
        encoded_sequences.append(encoded_seq)
    return np.array(encoded_sequences), aa_dict


def get_handcrafted_data(one_hot=False):
    train, test, _ = _get_data()
    _create_features(train)
    _create_features(test)
    _create_features_biopython(train)
    _create_features_biopython(test)
    _assertions(train)
    _assertions(test)
    _normalize(train, test)
    class_dict = _encode_class(train, one_hot=one_hot)
    x_train, y_train = _get_features(train)
    x_test, _ = _get_features(test)
    return x_train, y_train, x_test, class_dict


def get_handcrafted_raw_data():
    # For analysis purposes
    train, test, class_dict = _get_data()
    class_dict = _encode_class(train)
    _create_features(train, indiv_keys=True)
    _create_features(test, indiv_keys=True)
    _assertions(train)
    _assertions(test)
    return train, test, class_dict


def get_sequences(trim_len=2000, one_hot=True):
    train, test, max_len = _get_data()
    class_dict = _encode_class(train, one_hot=one_hot)
    x_train, aa_dict = _encode_aminoacids(train, one_hot=one_hot, pad=trim_len)
    x_test, _ = _encode_aminoacids(test, aa_dict, one_hot=one_hot, pad=trim_len)
    y_train = train['class']
    return x_train, y_train, x_test, class_dict

# get_data()

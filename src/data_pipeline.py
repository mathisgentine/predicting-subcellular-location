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
from utils import AA_CODES_LIST, AA_WILD, AA_WILD_LIST, AA_NAMES_LIST, aa_wild_prior, count_aa
from collections import Counter
import numpy as np

DATA_DIR = '../data'
CLASSES = ['cyto', 'mito', 'nucleus', 'secreted']
N_CLASSES = len(CLASSES)
TEST = 'blind'


def _parse_fasta(data, lines, c):
    lines_iter = iter(lines)
    line = next(lines_iter, None)
    while line:
        assert line.startswith('>')
        info = line
        line = next(lines_iter, None)
        seq = ''
        while line and not line.startswith('>'):
            seq += line.rstrip()
            line = next(lines_iter, None)
        data['info'].append(info)
        data['seq'].append(seq)
        data['class'].append(c)
    return data


def _load_data(data, name, c):
    with open('{}/{}.fasta'.format(DATA_DIR, name), 'r') as f:
        lines = f.readlines()
    _parse_fasta(data, lines, c)


def _get_data():
    train = {'info': [], 'seq': [], 'class': []}
    test = {'info': [], 'seq': [], 'class': []}
    for c in CLASSES:
        _load_data(train, c, c)
    _load_data(test, TEST, None)
    return train, test


def _assertions(data, eps=1e-7):
    # Check that assumptions made on the current dataset are correct
    for seq in data['seq']:
        assert '*' not in seq
        assert '-' not in seq

    # Feature assertions
    for global_rel, local_first_rel, local_last_rel in zip(data['global_rel'], data['local_first_rel'],
                                                           data['local_last_rel']):
        assert abs(1 - sum(global_rel)) < eps
        assert abs(1 - sum(local_first_rel)) < eps
        assert abs(1 - sum(local_last_rel)) < eps

    # Useful information
    c = Counter(data['class'])
    print('Class balance: {}'.format(c))


def _create_features(data, local_first=50, local_last=50):
    # Be careful: some chains have less than 50 aminoacids
    seq_lens = []
    global_rel = []
    local_first_rel = []
    local_last_rel = []
    mol_weights = []
    for seq in data['seq']:
        # Count aminoacids, find sequence length and compute molecular weight
        counts, seq_len, mol_weight = count_aa(seq)
        global_rel.append(counts)
        seq_lens.append(seq_len)
        mol_weights.append(mol_weight)

        # Count local_first first aminonacids
        counts, seq_len, _ = count_aa(seq[:local_first])
        local_first_rel.append(counts)

        # Count local_last last aminonacids
        counts, seq_len, _ = count_aa(seq[-local_last:])
        local_last_rel.append(counts)

    data['seq_len'] = np.array(seq_lens)
    data['global_rel'] = np.array(global_rel)
    data['molecular_weight'] = np.array(mol_weights)
    data['local_first_rel'] = np.array(local_first_rel)
    data['local_last_rel'] = np.array(local_last_rel)


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
    train['seq_len'], test['seq_len'] = _normalize_column(train['seq_len'], test['seq_len'])
    train['molecular_weight'], test['molecular_weight'] = _normalize_column(train['molecular_weight'],
                                                                            test['molecular_weight'])


def _get_features(data):
    x = np.concatenate((data['seq_len'][:, None],
                        data['global_rel'],
                        data['local_first_rel'],
                        data['local_last_rel'],
                        data['molecular_weight'][:, None]
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


def _encode_aminoacids(data, aa_dict=None, one_hot=True, pad=True):
    codes = AA_NAMES_LIST + ['X']
    if aa_dict is None:
        aa_dict = {c: i for i, c in enumerate(codes)}
    encoded_sequences = []
    for seq in data['seq']:
        encoded_seq = [aa_dict[aa] for aa in seq]
        if one_hot:
            encoded_seq = np.eye(len(codes))[encoded_seq]
        encoded_sequences.append(encoded_seq)
    return np.array(encoded_sequences), aa_dict


def get_handcrafted_data():
    train, test = _get_data()
    _create_features(train)
    _create_features(test)
    _assertions(train)
    _assertions(test)
    _normalize(train, test)
    class_dict = _encode_class(train)
    x_train, y_train = _get_features(train)
    x_test, _ = _get_features(test)
    return x_train, y_train, x_test, class_dict


def get_sequences():
    train, test = _get_data()
    class_dict = _encode_class(train)
    x_train, aa_dict = _encode_aminoacids(train)
    x_test, _ = _encode_aminoacids(test, aa_dict)
    y_train = train['class']
    return x_train, y_train, x_test, class_dict

# get_data()

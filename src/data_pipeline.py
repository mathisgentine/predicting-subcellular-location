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
    if 'global_rel' in data and 'local_first_rel' in data and 'local_last_rel' in data:
        for global_rel, local_first_rel, local_last_rel in zip(data['global_rel'], data['local_first_rel'],
                                                               data['local_last_rel']):
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
    global_rel = []
    local_first_rel = []
    local_last_rel = []
    mol_weights = []
    for seq in data['seq']:
        # Count aminoacids, find sequence length and compute molecular weight
        counts, seq_len, mol_weight, aa_list = count_aa(seq)
        global_rel.append(counts)
        seq_lens.append(seq_len)
        mol_weights.append(mol_weight)

        # Count local_first first aminonacids
        counts, seq_len, _, _ = count_aa(seq[:local])
        local_first_rel.append(counts)

        # Count local_last last aminonacids
        counts, seq_len, _, _ = count_aa(seq[-local:])
        local_last_rel.append(counts)

    global_rel = np.array(global_rel)
    local_first_rel = np.array(local_first_rel)
    local_last_rel = np.array(local_last_rel)

    data['seq_len'] = np.array(seq_lens)
    data['molecular_weight'] = np.array(mol_weights)
    if indiv_keys:
        for i, aa in enumerate(aa_list):
            data['global_rel_{}'.format(aa)] = global_rel[:, i]
            data['local_first_rel_{}'.format(aa)] = local_first_rel[:, i]
            data['local_last_rel_{}'.format(aa)] = local_last_rel[:, i]
    else:
        data['global_rel'] = global_rel
        data['local_first_rel'] = local_first_rel
        data['local_last_rel'] = local_last_rel
    return aa_list


def _create_features_biopython(data, local=50):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    mol_weights = []
    iso_points = []
    aromaticity = []
    gravy = []
    instability_index = []
    secondary_structure_fraction = []
    flexibility = []
    local_mw_first = []
    local_mw_last = []
    local_ip_first = []
    local_ip_last = []
    local_ssf_first = []
    local_ssf_last = []
    local_g_first = []
    local_g_last = []
    local_a_first = []
    local_a_last = []
    local_ii_first = []
    local_ii_last = []
    for seq in data['seq']:
        # Global features
        seq = replace_wild_first(seq)
        prot_analysis = ProteinAnalysis(seq)
        mol_weights.append(prot_analysis.molecular_weight())
        iso_points.append(prot_analysis.isoelectric_point())
        aromaticity.append(prot_analysis.aromaticity())
        secondary_structure_fraction.append(prot_analysis.secondary_structure_fraction())

        prot_analysis = ProteinAnalysis(replace_selenocysteine(seq))
        gravy.append(prot_analysis.gravy())
        instability_index.append(prot_analysis.instability_index())
        flexibility.append(flexibility_index(prot_analysis.flexibility()))

        # Local features
        prot_analysis = ProteinAnalysis(seq[:local])
        local_mw_first.append(prot_analysis.molecular_weight())
        local_ip_first.append(prot_analysis.isoelectric_point())
        local_ssf_first.append(prot_analysis.secondary_structure_fraction())
        local_a_first.append(prot_analysis.aromaticity())

        prot_analysis = ProteinAnalysis(replace_selenocysteine(seq[:local]))
        local_g_first.append(prot_analysis.gravy())
        local_ii_first.append(prot_analysis.instability_index())

        prot_analysis = ProteinAnalysis(seq[-local:])
        local_mw_last.append(prot_analysis.molecular_weight())
        local_ip_last.append(prot_analysis.isoelectric_point())
        local_ssf_last.append(prot_analysis.secondary_structure_fraction())
        local_a_last.append(prot_analysis.aromaticity())

        prot_analysis = ProteinAnalysis(replace_selenocysteine(seq[-local:]))
        local_g_last.append(prot_analysis.gravy())
        local_ii_last.append(prot_analysis.instability_index())

    data['molecular_weight'] = np.array(mol_weights)
    data['iso_point'] = np.array(iso_points)
    data['aromaticity'] = np.array(aromaticity)
    data['gravy'] = np.array(gravy)
    data['instability_index'] = np.array(instability_index)
    data['secondary_structure_fraction'] = np.array(secondary_structure_fraction)
    data['flexibility'] = np.array(flexibility)
    data['local_molecular_weight_first'] = np.array(local_mw_first)
    data['local_molecular_weight_last'] = np.array(local_mw_last)
    data['local_iso_point_first'] = np.array(local_ip_first)
    data['local_iso_point_last'] = np.array(local_ip_last)
    data['local_ssf_first'] = np.array(local_ssf_first)
    data['local_ssf_last'] = np.array(local_ssf_last)
    data['local_gravy_first'] = np.array(local_g_first)
    data['local_gravy_last'] = np.array(local_g_last)
    data['local_aromaticity_first'] = np.array(local_a_first)
    data['local_aromaticity_last'] = np.array(local_a_last)
    data['local_ii_first'] = np.array(local_ii_first)
    data['local_ii_last'] = np.array(local_ii_last)


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
    normalize_features = ['seq_len', 'molecular_weight', 'iso_point', 'gravy',
                          'aromaticity', 'instability_index', 'flexibility',
                          'local_molecular_weight_first', 'local_molecular_weight_last',
                          'local_iso_point_first', 'local_iso_point_last',
                          'local_gravy_first', 'local_gravy_last',
                          'local_aromaticity_first', 'local_aromaticity_last',
                          'local_ii_first', 'local_ii_last'
                          ]
    for feature in normalize_features:
        train[feature], test[feature] = _normalize_column(train[feature], test[feature])


def _get_features(data):
    x = np.concatenate((data['seq_len'][:, None],
                        data['global_rel'],
                        data['local_first_rel'],
                        data['local_last_rel'],
                        data['molecular_weight'][:, None],
                        data['iso_point'][:, None],
                        data['aromaticity'][:, None],
                        data['gravy'][:, None],
                        data['instability_index'][:, None],
                        data['secondary_structure_fraction'],
                        data['flexibility'][:, None],
                        data['local_molecular_weight_first'][:, None],
                        data['local_molecular_weight_last'][:, None],
                        data['local_iso_point_first'][:, None],
                        data['local_iso_point_last'][:, None],
                        data['local_ssf_first'],
                        data['local_ssf_last'],
                        data['local_gravy_first'][:, None],
                        data['local_gravy_last'][:, None],
                        data['local_aromaticity_first'][:, None],
                        data['local_aromaticity_last'][:, None],
                        #data['local_ii_first'][:, None],
                        #data['local_ii_last'][:, None]
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
            # raise NotImplementedError('Padding is only supported for one-hot vectors')
        encoded_sequences.append(encoded_seq)
    return np.array(encoded_sequences), aa_dict


def get_handcrafted_data():
    train, test, _ = _get_data()
    _create_features(train)
    _create_features(test)
    _create_features_biopython(train)
    _create_features_biopython(test)
    _assertions(train)
    _assertions(test)
    _normalize(train, test)
    class_dict = _encode_class(train)
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


def get_sequences(trim_len=2000, one_hot=False):
    train, test, max_len = _get_data()
    class_dict = _encode_class(train, one_hot=True)
    x_train, aa_dict = _encode_aminoacids(train, one_hot=one_hot, pad=trim_len)
    x_test, _ = _encode_aminoacids(test, aa_dict, one_hot=one_hot, pad=trim_len)
    y_train = train['class']
    return x_train, y_train, x_test, class_dict

# get_data()

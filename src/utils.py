"""
utils.py
Utilities
"""
from collections import Counter

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
    mol_weight = sum([v*AA_MOLECULAR_WEIGHTS[k] for k, v in aa_counts.items()])
    return rel_counts, seq_len, mol_weight

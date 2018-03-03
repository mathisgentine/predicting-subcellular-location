from data_pipeline import DATA_DIR
import numpy as np

BLOSUM_FILE = 'blosum62'


def _parse_blosum():
    with open('{}/{}.txt'.format(DATA_DIR, BLOSUM_FILE), 'r') as f:
        lines = f.readlines()

    blosum = {}
    lines_iter = iter(lines)
    line = next(lines_iter, None)
    aa_codes = line.split()
    line = next(lines_iter, None)
    while line:
        split = line.split()
        aa, values = split[0], split[1:]
        blosum[aa] = {k: int(v) for k, v in zip(aa_codes, values)}
        line = next(lines_iter, None)
    return blosum


def _fill_matrix(M, i, j, d, score, local):
    if i == 0 or j == 0 or not np.isnan(M[i, j]):
        return M[i, j]
    M_ = lambda i_, j_: _fill_matrix(M, i_, j_, d, score, local)
    M[i, j] = max((M_(i - 1, j - 1) + score(i, j),
                   M_(i, j - 1) - d,
                   M_(i - 1, j) - d))
    if local:
        M[i, j] = max((M[i, j], 0))
    return M[i, j]


def align_sequences(seq_1, seq_2, d, local):
    # Computes the alignment score of 2 sequences using the NW algorithm (global alignment) or
    # SW (local alignment)
    # with a BLOSUM scoring function
    # d: gap open penalty
    blosum = _parse_blosum()
    n, m = len(seq_1), len(seq_2)
    M = np.empty((n + 1, m + 1))
    M.fill(np.nan)
    M[:, 0] = -np.arange(n + 1) * d
    M[0, :] = -np.arange(m + 1) * d
    score = lambda i, j: blosum[seq_1[i-1]][seq_2[j-1]]  # 1-indexed
    ret = _fill_matrix(M, n, m, d, score, local)
    return ret


seq_1 = 'HGQKVADALTKAVAH'
seq_2 = 'VADALTKPVNFKFAVAH'
print(align_sequences(seq_1, seq_2, 0, local=True))

"""
score: auc=0.9857
"""

import pandas as pd
import numpy as np
import os
from glob import glob
from gru_framework import TOXIC_DATA_DIR


pd.options.display.max_columns = 6
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Controls weights when combining predictions
# 0: equal average of all inputs;
# 1: up to 50% of weight going to least correlated input
DENSITY_COEFF = 0.1
assert DENSITY_COEFF >= 0.0 and DENSITY_COEFF <= 1.0

# When merging 2 files with corr > OVER_CORR_CUTOFF
# the result's density is the max instead of the sum of the merged files' densities
OVER_CORR_CUTOFF = 0.98
assert OVER_CORR_CUTOFF >= 0.0 and OVER_CORR_CUTOFF <= 1.0

INPUT_DIR = 'submissions.ensemble'


def load_submissions():
    csv_files = glob(os.path.join(INPUT_DIR, '*.csv'))
    subs = {os.path.basename(path): pd.read_csv(path).sort_values('id') for path in csv_files}
    # subs = {path: pd.read_csv(path).sort_values('id') for path in csv_files}

    print('load_submissions: %s' % sorted(subs))
    return subs


load_submissions()
print('=' * 80)


def get_corr_mat(col, frames):
    c = pd.DataFrame()
    for name in sorted(frames):
        df = frames[name]
        c[name] = df[col]
    cor = c.corr()
    # print(cor)
    for name in cor.columns:
        cor.set_value(name, name, 0.0)
    return cor


def highest_corr(mat):
    n_cor = np.array(mat.values)
    corr = np.max(n_cor)
    idx = np.unravel_index(np.argmax(n_cor, axis=None), n_cor.shape)
    f1 = mat.columns[idx[0]]
    f2 = mat.columns[idx[1]]
    return corr, f1, f2


def get_merge_weights(m1, m2, densities):
    d1 = densities[m1]
    d2 = densities[m2]
    d_tot = d1 + d2
    weights1 = 0.5 * DENSITY_COEFF + (d1/d_tot)*(1-DENSITY_COEFF)
    weights2 = 0.5 * DENSITY_COEFF + (d2/d_tot)*(1-DENSITY_COEFF)
    return weights1, weights2


def ensemble_col(col, frames, densities):
    print('ensemble_col: col=%s frames=%d' % (col, len(frames)))
    if len(frames) == 1:
        _, fr = frames.popitem()
        return fr[col]

    mat = get_corr_mat(col, frames)
    corr, merge1, merge2 = highest_corr(mat)
    new_col_name = merge1 + '_' + merge2

    w1, w2 = get_merge_weights(merge1, merge2, densities)
    w1 += 0.1
    w2 = 1.0 - w1
    print('w1=%.3f w2=%.3f merge1=%s merge2=%s' % (w1, w2, merge1, merge2))
    new_df = pd.DataFrame()
    new_df[col] = (frames[merge1][col]*w1) + (frames[merge2][col]*w2)
    del frames[merge1]
    del frames[merge2]
    frames[new_col_name] = new_df

    if corr >= OVER_CORR_CUTOFF:
        print('\t', merge1, merge2, '  (OVER CORR)')
        densities[new_col_name] = max(densities[merge1], densities[merge2])
    else:
        print('\t', merge1, merge2)
        densities[new_col_name] = densities[merge1] + densities[merge2]

    del densities[merge1]
    del densities[merge2]
    # print(densities)
    return ensemble_col(col, frames, densities)


ens_submission = pd.read_csv(os.path.join(TOXIC_DATA_DIR, 'sample_submission.csv')).sort_values('id')
# ens_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv').sort_values('id')
# print(get_corr_mat('toxic',load_submissions()))

for col in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
    frames = load_submissions()
    print('\n\n', col)
    densities = {k: 1.0 for k in frames.keys()}
    ens_submission[col] = ensemble_col(col, frames, densities)

# print(ens_submission.describe())
OUTPUT = 'lazy_ensemble_submission2.csv'
ens_submission.to_csv(OUTPUT, index=False)
print('saved in %s' % OUTPUT)

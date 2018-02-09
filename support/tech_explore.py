# coding: utf-8
"""
    Explore the support data

    Available Data.xlsx crn.csv         orders.csv      tickets.csv
    all-tickets.txt     installs.csv        sql.txt


    installs: order_ref  !@#$

    'Satisfaction Score' == 'Bad'
        Tags
            17_2 apac jroxburgh_watch manual_bump mf pd_print_jobs_disappearing satisfaction_sent team_au
        Resolution time

        Compare categorical ratios: Bad vs Good

"""
import os
from os.path import expanduser, join, exists
import pandas as pd
from pprint import pprint
from utils import dim, compute_categoricals


pd.options.display.max_columns = 999
pd.options.display.width = 120

FORCE_READ = False
VERBOSE = True
SHOW_CATEGORICALS = True
GRAPHS = False

SHEET_NAMES = [
    'crn',
    'orders',
    'tickets',
    'installs',
]

DATA_DIR = expanduser('~/data/support-predictions/')
LOCAL_DATA_DIR = 'data.tech'

os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

sheets = {}
for name in SHEET_NAMES:
    local_path = join(LOCAL_DATA_DIR, '%s.pkl' % name)
    if not exists(local_path) or FORCE_READ:
        sheets[name] = pd.read_csv(join(DATA_DIR, '%s.csv' % name),
            encoding='latin-1', error_bad_lines=False)
        print('%20s: %s' % (name, dim(sheets[name])), flush=True)
        sheets[name].to_pickle(local_path)
    sheets[name] = pd.read_pickle(local_path)

for i, name in enumerate(SHEET_NAMES):
    print('%3d: %-15s %s' % (i, name, dim(sheets[name])), flush=True)

tickets = sheets['tickets']
print('tickets:', dim(tickets))
ID = 'Support ID [txt]'
CRN = 'Customer Reference Number (CRN) [txt]'
assert ID in tickets.columns, tickets.columns
assert CRN in tickets.columns, tickets.columns
for subset in [ID], [CRN], [ID, CRN]:
    df = tickets.dropna(subset=subset)
    print('drop:', dim(df), subset)

tickets = tickets.dropna(subset=[ID, CRN])
print('Satisfaction Score')
print(tickets['Satisfaction Score'].value_counts())
"""
    Satisfaction Score
    Not Offered    239154
    Offered          8095
    Good              940
    Bad                51
"""
bad = tickets[tickets['Satisfaction Score'] == 'Bad']
print('bad:', dim(bad))
print(bad[ID].value_counts())
print(bad[CRN].value_counts())
for k in ID, CRN:
    print('%-20s %d of %d' % (k, len(bad[k].unique()), len(bad)))


def normalize(s):
    s = s.lower()
    for w in ['unknown', 'SubReseller', 'n/a']:
        s = s.replace(w.lower(), '-')
    return s


id_crn = []
for _, row in bad.iterrows():
    s1 = normalize(row[ID])
    s2 = normalize(row[CRN])
    if s1 == '-' and s2 == '-':
        continue
    s = '%s::%s' % (s1, s2)
    id_crn.append(s)
print('both: %d of %d' % (len(set(id_crn)), len(id_crn)))
print(id_crn)
# bad.to_csv('bad.csv')
assert False

if VERBOSE:
    for name in sorted(sheets):
        print('=' * 80)
        df = sheets[name]
        print('%s: %s' % (name, list(df.shape)))
        for i, col in enumerate(df.columns):
            print('%6d: %s' % (i, col))

    if False:
        for name in sorted(sheets):
            print('-' * 80)
            df = sheets[name]
            print('%s: %s' % (name, list(df.shape)))
            print(sheets[name].head())

    if False:
        for name in sorted(sheets):
            print('^' * 80)
            df = sheets[name]
            print('%s: %s' % (name, list(df.shape)))
            print(sheets[name].describe())


if False:
    scores = sheets['tickets']['Satisfaction Score']
    levels = sorted(set(scores))
    print(len(scores))
    print(levels)
    for l in levels:
        m = len([v for v in scores if v == l])
        print('%20s: %7d %.3f' % (l, m, m / len(scores)))


def show_categoricals(name, df, threshold=20):
    print('=' * 80)
    print(name)
    categoricals = []
    for i, col in enumerate(df.columns):
        scores = df[col]
        try:
            levels = list(set(scores))
        except Exception as e:
             print('%4d: %-20s - %s' % (i, col, e))
             continue

        level_counts = {l: len([v for v in scores if v == l]) for l in levels}
        levels.sort(key=lambda l: (-level_counts[l], l))
        print('%4d: %-20s - %7d levels %7d total %4.1f%%' % (i, col, len(levels), len(scores),
            100.0 * len(levels) / len(scores)))
        if len(levels) <= threshold:
            categoricals.append(col)
            for l in levels:
                m = level_counts[l]
                print('%20s: %7d %4.1f' % (l, m, 100.0 * m / len(scores)), flush=True)
    print('-' * 80)
    print('categoricals: %d %s' % (len(categoricals), categoricals))
    return categoricals


if SHOW_CATEGORICALS:
    name_categoricals = {}
    for name in SHEET_NAMES:
        name_categoricals[name] = compute_categoricals(name, sheets[name], threshold=20)
    pprint(name_categoricals)
    assert False

if False:
    orders = sheets['orders']
    money_columns = [
        'value_rrp_aud',        # 285850 levels 521656 total
        'value_native',         #  42853 levels 521656 total
        'quantity',             #   1187 levels 521656 total
        'value_aud',            # 330148 levels 521656 total
        'value_native_rrp',     #  21352 levels 521656 total
        ]

    print(orders[money_columns].describe())
    for col in money_columns:
        print('%-20s: %.4g' % (col, orders[col].sum()))

# coding: utf-8
"""
    Explore the support data


    Available Data.xlsx crn.csv         orders.csv      tickets.csv
    all-tickets.txt     installs.csv        sql.txt

"""
from os.path import expanduser, join
import pandas as pd


pd.options.display.max_columns = 999
pd.options.display.width = 120

VERBOSE = True
CATEGORICALS = True
GRAPHS = False

files = [
    # 'crn',
    # 'orders',
    # 'tickets',
    # 'installs',
    'quote-order',
    ]

data_dir = expanduser('~/data/support-predictions/')

sheets = {}
for name in files:
    sheets[name] = pd.read_csv(join(data_dir, '%s.csv' % name),
        encoding='latin-1', error_bad_lines=False)
    print('%20s: %s' % (name, list(sheets[name].shape)))

# crn = pd.read_csv(join(data_dir, 'crn.csv'), encoding='latin-1')
# orders = pd.read_csv(join(data_dir, 'orders.csv'))
# tickets = pd.read_csv(join(data_dir, 'tickets.csv'))
# installs = pd.read_csv(join(data_dir, 'installs.csv'))

# sheets = {
#     'crn': crn,
#     'orders': orders,
#     'tickets': tickets,
#     'installs': installs
# }

if VERBOSE:
    for name in sorted(sheets):
        print('=' * 80)
        df = sheets[name]
        print('%s: %s' % (name, list(df.shape)))
        for i, col in enumerate(df.columns):
            print('%6d: %s' % (i, col))

    for name in sorted(sheets):
        print('-' * 80)
        df = sheets[name]
        print('%s: %s' % (name, list(df.shape)))
        print(sheets[name].head())

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


if CATEGORICALS:
    name_categoricals = {}
    for name in files:
        name_categoricals[name] = show_categoricals(name, sheets[name])

    print('#' * 80)
    for name in files:
        categoricals = name_categoricals[name]
        print('%-20s: %d %s' % (name, len(categoricals), categoricals))


if False:
    orders = sheets['orders']
    money_columns = [
        'value_rrp_aud',        #- 285850 levels 521656 total
        'value_native',         #-  42853 levels 521656 total
        'quantity',             #-   1187 levels 521656 total
        'value_aud',            #- 330148 levels 521656 total
        'value_native_rrp',     #-  21352 levels 521656 total
        ]


    print(orders[money_columns].describe())
    for col in money_columns:
        print('%-20s: %.4g' % (col, orders[col].sum()))

# coding: utf-8
"""
    Explore the sales data

"""
from os.path import expanduser, join, exists
from collections import defaultdict
from pprint import pprint
import numpy as np
import pandas as pd
from utils import save_json, load_json


pd.options.display.max_columns = 999
pd.options.display.width = 120

FORCE_READ = False
VERBOSE = False
APPLY_CATEGORICALS = True
MAKE_CATEGORICALS = False
GRAPHS = False
QUOTE_SALE = False

name = 'quote-order'
local_path = '%s.pkl' % name
data_dir = expanduser('~/data/support-predictions/')


def convert_categoricals(df):
    enumerations = load_json('enumerations.json')
    df2 = pd.DataFrame(index=df.index)
    for col in df.columns:
        enum = enumerations.get(col)
        if not enum:
            df2[col] = df[col]
            continue
        convert = {v: i for i, v in enumerate(enum)}
        # trevnoc = {i: v for i, v in enumerate(enum)}
        v_nan = convert.get('nan')
        vals = list(df[col])
        print('Convert %r : %d' % (col, len(convert)))
        if v_nan is None:
            for v in vals:
                assert v in convert, (v, enum[:10])
        categorical = [convert.get(v, v_nan) for v in vals]
        print(len(categorical), sorted(set(categorical)))
        # df2[col] = pd.Series(categorical, dtype=np.int32).astype(np.int32)
        df2[col] = pd.Series(categorical, index=df.index)
        print(len(df2[col]), sorted(set(df2[col].values)))
        df2[col] = df2[col].astype(np.int32)
        print('%20s : %20s -> %20s' % (col, df[col].dtype, df2[col].dtype))
    return df2


if not exists(local_path) or FORCE_READ:
    col_dtype = {
        'ASC': str,
        'Manufacturer': str,
        'Super Region': str,
        'Parent Region': str,
        'Reseller Region': str,
        'Reseller Tier': str,
        'Reseller Code': str,
        'Reseller Name': str,
        'Reseller City': str,
        'Reseller State': str,
        'Reseller Country': str,
        'Reseller Enabled/Disabled': str,
        'Customer Organisation': str,
        'Customer City': str,
        'Customer Country': str,
        'id': np.float64,  # np.int32,
        'address1': str,
        'address2': str,
        'city': str,
        'country': str,
        'state': str,
        'zip': str,
        'firstName': str,
        'lastName': str,
        'organizationName': str,
        'clientPurchaseOrderNumber': str,
        'currency': str,
        'dateCreated': str,
        'electedPaymentMethod': str,
        'isQuote': np.float64,  # np.int32,
        'orderNumber': np.float64,  # np.int32,
        'orderProductLine': str,
        'orderStatus': str,
        'orderSystemEntryMethod': str,
        'orderSystemIpAddress': str,
        'orgType': str,
        'discountPercentage': np.float64,
        'discountReason': str,
        'discountType': str,
        'paymentSecurityId': np.float64,  # np.int32,
        'paymentStatus': str,
        'priceAtOrderTime': np.float64,
        'quoteNumber': str,
        'resellerCode': str,
        'resellerDiscountPercentage': str,
        'specialOrderNumber': str,
        'subreseller': str,
        'subSubReseller': str,
        'type': str,
        'customerId': np.float64,  # np.int32,
        'customerComment': str,
        'exchangeRateAtOrderTime': str,
        'resellerDiscountReason': str,
        'emailAddress': str,
        'exportedToNetSuite': str,
        'endCustomerId': np.float64,  # np.int32,
        'originalEndCustomerId': np.float64,  # np.int32,
    }
    int_cols = ['id', 'isQuote',
                # 'orderNumber',
                 # 'paymentSecurityId',
                 'customerId']
    bad_int_cols = ['endCustomerId',
                'originalEndCustomerId']
    single_val_cols = ['Reseller Enabled/Disabled', 'exportedToNetSuite']

    orders = pd.read_csv(join(data_dir, '%s.csv' % name),
                         # dtype=col_dtype,
                         parse_dates=['dateCreated'],
                         encoding='latin-1', error_bad_lines=False)
    print('%s: %s' % (name, list(orders.shape)))
    for col in int_cols:
        print(col, end=' ', flush=True)
        orders = orders[pd.notnull(orders[col])]
        print(list(orders.shape), end=' ', flush=True)
        orders[col] = orders[col].astype(np.int32)
        print(' !')
    for col in bad_int_cols:
        orders[col] = orders[col].fillna(-1.0)
        orders[col] = orders[col].astype(np.int32)
        print('%s: %s' % (name, list(orders.shape)))
    orders['discountPercentage'] = orders['discountPercentage'].fillna(0.0)
    good_cols = [col for col in orders.columns if col not in single_val_cols]
    orders = orders[good_cols]

    # orders = orders[pd.notnull(orders[int_cols])]
    # orders[int_cols] = orders[int_cols].astype(np.int32)
    orders.sort_values(by=['dateCreated'], inplace=True)
    orders.to_pickle(local_path)

orders = pd.read_pickle(local_path)
print('%s: %s' % (name, list(orders.shape)))

if not MAKE_CATEGORICALS:
    orders = convert_categoricals(orders)
    for col in orders.columns:
        print('%20s : %s' % (col, orders[col].dtype))

    if APPLY_CATEGORICALS:
        orders.to_pickle(local_path)


if VERBOSE:
    print('=' * 80)
    for col in orders.columns:
        print("'%s': %s" % (col, orders[col].dtype))

    print('=' * 80)
    df = orders
    print('%s: %s' % (name, list(df.shape)))
    for i, col in enumerate(df.columns):
        print('%6d: %s' % (i, col))

    print('-' * 80)
    df = orders
    print('%s: %s' % (name, list(df.shape)))
    print(orders.head())

    print('^' * 80)
    df = orders
    print('%s: %s' % (name, list(df.shape)))
    print(orders.describe())


def show_categoricals(name, df, threshold=20):
    print('=' * 80)
    print('categorical: %s' % name)
    print(list(df.columns))
    categoricals = []
    single_val_cols = []
    enumerations = {}

    for i, col in enumerate(df.columns):
        scores = df[col]
        print('%4d: %-20s %-10s - ' % (i, col, scores.dtype), end='', flush=True)
        is_object = str(scores.dtype) == 'object'
        scores = list(scores)
        print('; ', end='', flush=True)
        if is_object:
            scores = [str(v) for v in scores]
        print(', ', end='', flush=True)

        try:
            levels = set(scores)
            print('. ', end='', flush=True)
            levels = list(levels)
        except Exception as e:
             print(e)
             continue
        print('%7d levels %7d total %4.1f%%' % (len(levels), len(scores),
            100.0 * len(levels) / len(scores)), end='', flush=True)

        if len(levels) == 1:
            single_val_cols.append(col)

        if len(levels) <= threshold:
            level_counts = {l: len([v for v in scores if v == l]) for l in levels}
            print(' ;', end='', flush=True)

            levels.sort(key=lambda l: (-level_counts[l], l))
            enumerations[col] = levels

        print(' ***', flush=True)

        if len(levels) <= threshold:
            categoricals.append(col)
            for l in levels:
                m = level_counts[l]
                print('%20s: %7d %4.1f' % (l, m, 100.0 * m / len(scores)), flush=True)
    print('-' * 80)
    print('categoricals: %d %s' % (len(categoricals), categoricals))
    return categoricals, single_val_cols, enumerations


if MAKE_CATEGORICALS:
    _, single_val_cols, enumerations = show_categoricals(name, orders, threshold=100)
    print('single_val_cols=%d %s' % (len(single_val_cols), single_val_cols))
    save_json('enumerations.json', enumerations)
    assert False

customer_col = 'emailAddress'
customer_col = 'Customer Organisation'
# customer_col = 'customerId'
# customer_col = 'endCustomerId'
# customer_col = 'clientPurchaseOrderNumber'
# customer_col = 'electedPaymentMethod'


customers = orders[customer_col]
unique_customers = set(customers)
unique_types = {type(cust) for cust in customers}
float_customers = sorted({cust for cust in customers if isinstance(cust, float)})

print('-' * 80)
print('customer_col=%r' % customer_col)
print('customers=%d unique_customers=%d' % (len(customers), len(unique_customers)))
print('types=%s' % unique_types)
print('float_customers=%d %s' % (len(float_customers), float_customers[:10]))

unique_customers = sorted({cust for cust in customers if isinstance(cust, str)})
print('unique_customers=%d %s' % (len(unique_customers), unique_customers[:10]))

if False:
    quote_counts = defaultdict(int)
    for i, cust in enumerate(unique_customers):
        both = orders.loc[orders[customer_col] == cust]
        quotes = both.loc[orders['isQuote'] == 0.0]
        sales = both.loc[orders['isQuote'] == 1.0]
        b, q, s = [len(df) for df in (both, quotes, sales)]
        if b == 0:
            continue
        assert b == q + s, (i, cust, b, q, s)
        # assert b > 0, (i, cust, b, q, s)
        quote_counts[(q > 0, s > 0)] += 1
        if i % 1000 == 1 and i <= 10001:
            pprint({k: v for k, v in quote_counts.items()})
        if q >= 4 and s == 1:
            both_name = 'both_%s_%d_%d.csv' % (customer_col, q, s)
            print('Saving to %r' % both_name, both.shape)
            both.to_csv(both_name, index=False)
            break

    pprint({k: v for k, v in quote_counts.items()})

    # customer_col = 'Customer Organisation'
    # unique_customers=35026
    # {(False, True): 403, (True, False): 27524, (True, True): 7099}

if QUOTE_SALE:
    # 'priceAtOrderTime' same for quote and order. order later than quote

    quote_sales = defaultdict(list)
    show_len = -1

    for i, cust in enumerate(unique_customers):
        customer_orders = orders.loc[orders[customer_col] == cust]
        quotes = customer_orders.loc[orders['isQuote'] == 1.0]
        if len(quotes) == 0:
            continue
        # print('quotes=%s' % type(quotes), quotes.shape)
        for k, quote in quotes.iterrows():
            # print('quote=%s %s' % (type(quote), [type(v) for v in quote]))
            # k, v = quote
            # print('k=%d v=%s' % (k, list(v.shape)))
            price = quote['priceAtOrderTime']
            date = quote['dateCreated']
            sales = customer_orders.loc[
                                (orders['isQuote'] == 0.0) &
                                (orders['priceAtOrderTime'] == price) &
                                (orders['dateCreated'] >= date)
                              ]
            oid = quote['id']
            assert oid not in quote_sales, oid
            if len(sales) == 0:
                continue

            # print('sales=%s' % type(sales), sales.shape)
            quote_sales[oid] = [s['id'] for _, s in sales.iterrows()]
            if len(quote_sales) % 10 == 1 and show_len < len(quote_sales) <= 21:
                print('%6d of %d customers. %6d converted quotes (%4.1f%%)' % (
                    i, len(unique_customers), len(quote_sales),
                    100.0 * len(quote_sales) / len(unique_customers)), flush=True)
                pprint({k: v for k, v in quote_sales.items()})
                show_len = len(quote_sales)
        if i % (len(unique_customers) // 100) == len(unique_customers) // 200:
            print('%6d of %d customers. %6d converted quotes (%4.1f%%)' % (
                i, len(unique_customers), len(quote_sales),
                100.0 * len(quote_sales) / len(unique_customers)), flush=True)

    print('%d converted quotes from %d customers' % (len(quote_sales), i), flush=True)
    # pprint({k: v for k, v in quote_sales.items()})
    save_json('converted_quotes.json', quote_sales)


print('Done **********')

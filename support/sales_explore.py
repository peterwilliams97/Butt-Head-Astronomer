# coding: utf-8
"""
    Explore the sales data

"""
from os.path import join, exists
from collections import defaultdict
from pprint import pprint
import numpy as np
import pandas as pd
from datetime import timedelta
from utils import (save_json, load_json, ORDERS_NAME, ORDERS_LOCAL_PATH, DATA_DIR,
    convert_categoricals, compute_categoricals)


pd.options.display.max_columns = 999
pd.options.display.width = 120

FORCE_READ = False
VERBOSE = False
APPLY_CATEGORICALS = False
MAKE_CATEGORICALS = False
SHOW_CATEGORICALS = False
GRAPHS = False
QUOTE_SALE = False

assert not (MAKE_CATEGORICALS and APPLY_CATEGORICALS)

CATEGORY_ENUMERATIONS = 'enumerations.json'
CONVERTED_QUOTES = 'converted_quotes.json'


if not exists(ORDERS_LOCAL_PATH) or FORCE_READ or APPLY_CATEGORICALS:
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
    int_cols = ['id', 'isQuote', 'customerId']
    bad_int_cols = ['endCustomerId', 'originalEndCustomerId']
    single_val_cols = ['Reseller Enabled/Disabled', 'exportedToNetSuite']

    orders = pd.read_csv(join(DATA_DIR, '%s.csv' % ORDERS_NAME), parse_dates=['dateCreated'],
                         encoding='latin-1', error_bad_lines=False)
    print('%s: %s' % (ORDERS_NAME, list(orders.shape)))
    for col in int_cols:
        print(col, end=' ', flush=True)
        orders = orders[pd.notnull(orders[col])]
        print(list(orders.shape), end=' ', flush=True)
        orders[col] = orders[col].astype(np.int32)
        print(' !')
    for col in bad_int_cols:
        orders[col] = orders[col].fillna(-1.0)
        orders[col] = orders[col].astype(np.int32)
        print('%s: %s' % (ORDERS_NAME, list(orders.shape)))
    orders['discountPercentage'] = orders['discountPercentage'].fillna(0.0)
    good_cols = [col for col in orders.columns if col not in single_val_cols]
    orders = orders[good_cols]

    orders.sort_values(by=['dateCreated'], inplace=True)
    orders.to_pickle(ORDERS_LOCAL_PATH)

orders = pd.read_pickle(ORDERS_LOCAL_PATH)
print('%s: %s' % (ORDERS_NAME, list(orders.shape)))

if APPLY_CATEGORICALS:
    enumerations = load_json(CATEGORY_ENUMERATIONS)
    orders = convert_categoricals(orders, enumerations)
    for col in orders.columns:
        print('%20s : %s' % (col, orders[col].dtype))
    orders.to_pickle(ORDERS_LOCAL_PATH)
    assert False

if VERBOSE:
    print('=' * 80)
    for col in orders.columns:
        print("'%s': %s" % (col, orders[col].dtype))

    print('=' * 80)
    df = orders
    print('%s: %s' % (ORDERS_NAME, list(df.shape)))
    for i, col in enumerate(df.columns):
        print('%6d: %s' % (i, col))

    print('-' * 80)
    df = orders
    print('%s: %s' % (ORDERS_NAME, list(df.shape)))
    print(orders.head())

    print('^' * 80)
    df = orders
    print('%s: %s' % (ORDERS_NAME, list(df.shape)))
    print(orders.describe())

if MAKE_CATEGORICALS or SHOW_CATEGORICALS:
    threshold = 1000 if MAKE_CATEGORICALS else 20
    col_level, _, single_val_cols, enumerations = compute_categoricals(ORDERS_NAME, orders,
        threshold=threshold)
    for i, col in enumerate(sorted(col_level, key=lambda k: (-col_level[k], k))):
        print('%3d: %30s %6d %5.1f%%' % (i, col, col_level[col],
              100.0 * col_level[col] / len(orders)))
    print('col_level_type = [')
    for i, col in enumerate(sorted(col_level, key=lambda k: (-col_level[k], k))):
        print('    (%30r, %6d, %8s),  # %3d  %5.1f%%' % (
              col, col_level[col], orders[col].dtype,
              i, 100.0 * col_level[col] / len(orders)
              ))
    print(']')
    print('single_val_cols=%d %s' % (len(single_val_cols), single_val_cols))
    if MAKE_CATEGORICALS:
        save_json(CATEGORY_ENUMERATIONS, enumerations)
        assert False

if QUOTE_SALE:
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
        quotes = both.loc[orders['isQuote'] == 0]
        sales = both.loc[orders['isQuote'] == 1]
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

    customer_quote_all_sales = []
    quote_duration = timedelta(days=180)

    for i, cust in enumerate(unique_customers):
        quote_all_sales = []
        customer_orders = orders.loc[orders[customer_col] == cust]
        quotes = customer_orders.loc[orders['isQuote'] == 1]
        if len(quotes) == 0:
            continue
        for k, quote in quotes.iterrows():
            price = quote['priceAtOrderTime']
            date = quote['dateCreated']
            all_sales = customer_orders.loc[(orders['isQuote'] == 0) &
                                            (orders['dateCreated'] >= date)]
            if len(all_sales) == 0:
                continue
            ss = [(s['dateCreated'].strftime('%Y-%m-%d'), s['priceAtOrderTime'])
                  for _, s in all_sales.iterrows()]
            quote_all_sales.append(((date.strftime('%Y-%m-%d'), price), ss))

            expiry = date + quote_duration
            # sales = all_sales.loc[orders['priceAtOrderTime'] == price]
            sales = all_sales.loc[orders['dateCreated'] <= expiry]
            oid = quote['id']
            assert oid not in quote_sales, oid
            if len(sales) == 0:
                continue

            quote_sales[oid] = [s['id'] for _, s in sales.iterrows()]
            if len(quote_sales) % 10 == 1 and show_len < len(quote_sales) <= 21:
                print('%6d of %d customers. %6d converted quotes (%4.1f%%)' % (
                    i, len(unique_customers), len(quote_sales),
                    100.0 * len(quote_sales) / len(unique_customers)), flush=True)
                pprint({k: v for k, v in quote_sales.items()})
                show_len = len(quote_sales)
            if quote_all_sales:
                customer_quote_all_sales.append((cust, quote_all_sales))

        if i % (len(unique_customers) // 100) == max(1, len(unique_customers) // 1000):
            print('%6d of %d customers. %6d converted quotes (%4.1f%%)' % (
                i, len(unique_customers), len(quote_sales),
                100.0 * len(quote_sales) / len(unique_customers)), flush=True)
        # if len(customer_quote_all_sales) >= 1000:
        #     break

    print('%d converted quotes from %d customers' % (len(quote_sales), i), flush=True)
    # pprint({k: v for k, v in quote_sales.items()})
    converted_quotes = [(q, quote_sales[q]) for q in sorted(quote_sales)]
    save_json(CONVERTED_QUOTES, converted_quotes)

    for k, (cust, quote_all_sales) in enumerate(customer_quote_all_sales):
        print('%4d: %s' % (k, '=' * 74))
        print('customer: %r' % cust)
        for i, (quote_price, sales_prices) in enumerate(quote_all_sales):
            print('%4d: %s' % (i, '-' * 74))
            print('%4s: %s %d sales' % ('quote', quote_price, len(sales_prices)))
            matched = False
            for j, sp in enumerate(sales_prices):
                marker = ''
                if not matched:
                    if sp[1] == quote_price[1]:
                        matched = True
                        marker = 'match'
                print('%5d: %s %s' % (j, sp, marker))

    save_json('customer_quote_all_sales.json', customer_quote_all_sales)
    assert False

if True:
    import re

    print('-' * 80)
    print('isQuote values:', flush=True)
    print(orders['isQuote'].value_counts().iloc[:10])
    print('-' * 80)
    print('Customer Organisation values:', flush=True)
    print(orders['Customer Organisation'].value_counts().iloc[:10])
    print('quoteNumber:', flush=True)
    print(orders['quoteNumber'].value_counts().iloc[:10])

    qns = orders['quoteNumber'].fillna('XXXX-XX').values
    print('qns=%s %s %s' % (type(qns), list(qns.shape), qns.dtype))

    re_qns = re.compile(r'^Q(\d+)$')
    vals = []
    n_empty = 0
    for i, q in enumerate(qns):
        if q == 'XXXX-XX':
            n_empty += 1
        else:
            m = re_qns.search(str(q))
            assert m, (i, q)
            n = int(m.group(1))
            vals.append(n)
    print(n_empty, len(orders), len(orders) - n_empty, n_empty / len(orders))
    print(min(vals))
    print(max(vals))

print('Done **********')

# from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import pandas as pd
from utils import orders_name, local_path, load_json


orders = pd.read_pickle(local_path)
print('%s: %s' % (orders_name, list(orders.shape)))

Xcols = [col for col in orders.columns if orders[col].dtype == np.int32]
print(len(Xcols))
orders = orders[Xcols]
print('%s: %s' % (orders_name, list(orders.shape)))

converted_quotes = load_json('converted_quotes.json')
converted = {q for q, s in converted_quotes if s}

quotes = orders[orders['isQuote'] == 1]
print('%s: %s' % ('quotes', list(quotes.shape)))
quotes = quotes[[col for col in quotes.columns if col not in
           {'endCustomerId', 'originalEndCustomerId'}]]
print('%s: %s' % ('quotes', list(quotes.shape)))

y = pd.DataFrame(index=quotes.index)
y_values = [i in converted for i in quotes['id'].values]
y['converted'] = pd.Series(y_values, index=quotes.index, dtype=np.int32)
print('y', y.shape, y['converted'].dtype)

X = quotes[[col for col in quotes.columns if col == 'isQuote']]
print('X', X.shape)

# print(X.describe())

clf = SelectKBest(chi2, k=2).fit(X, y)
X_new = clf.transform(X)
support = clf.get_support(indices=True)
columns = [X.columns[i] for i in support]
print('X_new', X_new.shape, support, columns)

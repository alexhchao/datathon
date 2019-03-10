
#############
# implement your own factor model

# sept 22, 2018
##############
from scipy.stats import mstats

# 3:43

import pandas as pd
import numpy as np
import os
import sys
import matplotlib as plt

print('hello')

os.getcwd()

raw_data = pd.read_csv('db_data.csv')

raw_data.describe()

raw_data.date.unique

pd.options.display.max_rows = 15
pd.options.display.max_columns = 15

raw_data

##################

# 1) let us proxy Delta inverse by using sqrt(mktcap)
# create FMPs
##################
# Winsorize at -3, +3

factor_mapping = {'F01':'sector',
                  'F03': 'mktcap',
                  'F05': 'momentum',
                  'F09': 'quality',
                  'F12': 'growth',
                  'F13': 'vol',
                  'F15': 'value'}

factor_mapping

raw_data.set_index(['stock','date'], inplace = True)

X_raw = raw_data.loc[:, list(factor_mapping.keys()) + ['fwd_returns']].copy()

X_raw.rename(columns = factor_mapping, inplace = True)

X_raw


fwd_returns = raw_data.fwd_returns * 0.01
returns = raw_data.fwd_returns.groupby('stock').shift(1) * 0.01

X_raw['returns'] = X_raw.fwd_returns.groupby('stock').shift(1)

returns.unstack().T

fwd_returns.unstack().T

y = returns.copy()

##########

_x = np.random.randn(1000)

pd.Series(_x).hist()

plt.show(block=True)

import matplotlib.pyplot as plt

plt.interactive(True)

#plt.show()

_x_ws = mstats.winsorize(_x, limits = [0.1, .1])

pd.Series(_x_ws).hist()

X_raw.describe()

X_d1 = X_raw[X_raw.index.get_level_values('date')=='D001']

X_d1.describe()

##############
# winsorize
X_d1_ws = X_d1.apply(lambda x: mstats.winsorize(x, limits = [.01,.01]))

y_d1 = y.loc[y.index.get_level_values('date') == 'D001']

y_d1.dropna()

##############
this_date = 'D001'

X_this_date = X_raw[X_raw.index.get_level_values('date')==this_date]

# aggressive, but drop rows where all is na
_X = X_this_date.dropna(axis=0, how='any')

_X.index = _X.index.droplevel(1)


_X_sector_dummies = pd.concat([_X,pd.get_dummies(_X.sector)],axis=1)
_X_sector_dummies = _X_sector_dummies.iloc[:,1:]


sqrt_mktcap = np.sqrt(_X_sector_dummies.mktcap)

DELTA = pd.DataFrame(np.diag(sqrt_mktcap))
# n by n

# exclude mktcap from X
returns_this_date = _X_sector_dummies.loc[:,['returns']]

_X_ready = _X_sector_dummies.loc[:,[c for c in _X_sector_dummies.columns if c not in ['mktcap',
                                                                           'fwd_returns',
                                                                           'returns']]]

# need to be z-scored
_X_ready_z = _X_ready.iloc[:,:5].apply(zscore, axis=0)

_X_read_z_sectors = pd.concat([_X_ready_z, pd.get_dummies(_X.sector)],axis=1)

def zscore(x):
    return (x - x.mean()) / x.std()


DELTA_m = np.matrix(DELTA)
_X_m = np.matrix(_X_read_z_sectors )

fmp = np.linalg.inv(_X_m.T.dot(DELTA_m).dot(_X_m)).dot(_X_m.T).dot(DELTA_m)

fmp_df = pd.DataFrame(fmp )

fmp_df.sum(axis=1)
fmp_df.T.describe()

fmp_df.iloc[13].hist(bins=100)
fmp_df.index = _X_ready.columns

pd.DataFrame(_X_m).mean()

DELTA.dot(DELTA)

q = np.matrix([1,2,3,4]).reshape(2,2)

p = np.matrix([6,7,8,9]).reshape(2,2)

q.dot(p)


##############

# 2)
#
##############
fmp_df.T.iloc[:,:5].corr()

factor_returns = np.matrix(fmp_df).dot(np.matrix(returns_this_date/100))
factor_returns_df = pd.DataFrame(factor_returns)
factor_returns_df.index= _X_ready.columns

factor_returns_df

all_days = X_raw.reset_index().date.unique()
all_days = all_days[1:]


all_factor_rets = []
all_fmps = {}

for this_date in all_days:
    print(this_date)
    X_this_date = X_raw[X_raw.index.get_level_values('date') == this_date]

    # aggressive, but drop rows where all is na
    _X = X_this_date.dropna(axis=0, how='any')

    _X.index = _X.index.droplevel(1)

    _X_sector_dummies = pd.concat([_X, pd.get_dummies(_X.sector)], axis=1)
    _X_sector_dummies = _X_sector_dummies.iloc[:, 1:]

    sqrt_mktcap = np.sqrt(_X_sector_dummies.mktcap)

    DELTA = pd.DataFrame(np.diag(sqrt_mktcap))
    # n by n
    # exclude mktcap from X
    returns_this_date = _X_sector_dummies.loc[:, ['returns']]

    _X_ready = _X_sector_dummies.loc[:, [c for c in _X_sector_dummies.columns if c not in ['mktcap',
                                                                                           'fwd_returns',
                                                                                           'returns']]]

    # need to be z-scored
    _X_ready_z = _X_ready.iloc[:, :5].apply(zscore, axis=0)

    _X_read_z_sectors = pd.concat([_X_ready_z, pd.get_dummies(_X.sector)], axis=1)


    DELTA_m = np.matrix(DELTA)
    _X_m = np.matrix(_X_read_z_sectors)

    fmp = np.linalg.inv(_X_m.T.dot(DELTA_m).dot(_X_m)).dot(_X_m.T).dot(DELTA_m)

    fmp_df = pd.DataFrame(fmp)

    fmp_df.sum(axis=1)
    #fmp_df.T.describe()

    #fmp_df.iloc[13].hist(bins=100)
    fmp_df.index = _X_ready.columns
    fmp_df.columns = sqrt_mktcap.index

    print(fmp_df.head())

    all_fmps[this_date] = fmp_df.T

    #############
    # now calc factor returns

    factor_returns = np.matrix(fmp_df).dot(np.matrix(returns_this_date / 100))
    factor_returns_df = pd.DataFrame(factor_returns)
    factor_returns_df.index = _X_ready.columns

    factor_returns_df.columns = [this_date]

    print(factor_returns_df)
    all_factor_rets.append(factor_returns_df)


all_fmps

all_factor_rets_df = pd.concat(all_factor_rets, axis=1)


np.cumprod(1+all_factor_rets_df.T.iloc[:,:5]).plot()


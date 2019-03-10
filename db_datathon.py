# 3:43

import pandas as pd
import numpy as np
import os
import sys

print('hello')

os.getcwd()

raw_data = pd.read_csv('db_data.csv')

raw_data.describe()

raw_data.date.unique

pd.options.display.max_rows = 15
pd.options.display.max_columns = 15

#############################
# exploratory analysis - outliers, missing values
#############################

raw_data.isnull().sum()/raw_data.shape[0]

raw_data.describe()

#############################
# lets run panel regressions
#############################
import statsmodels.api as sm

X = raw_data.set_index(['stock','date']).loc[:,'F01':'F20']

unique_vals = {k:raw_data[k].unique()[:10] for k in raw_data.columns}
# F02 is sector

ranks = raw_data.groupby('date').rank()

ranks = ranks[ranks.fwd_returns.notnull()]

ranks_drop_all = ranks.dropna()

y = ranks_drop_all.fwd_returns

X = ranks_drop_all.loc[:,[c for c in ranks_drop_all.columns if c not in ['fwd_returns','stock']]]


model = sm.OLS(endog = y,exog = X).fit()

model.summary()

#############################
# lets try running a rolling window lasso
#############################






#############################
# lets split first 80 as training set

raw_data.loc[raw_data.date == 'D080','date']

# index = 87680
# drop F02 for now (sectors)
cols = [c for c in raw_data.columns if c != 'F02']

train = raw_data.loc[:87680,cols]
test  = raw_data.loc[87681:,cols]


# training set
# ============

# lets convert all factors to ranks
cols = [c for c in train.columns if c not in ['stock']]
factors = ['F'+str(i).zfill(2) for i in range(21)]

ranks = train.loc[:,cols].groupby('date').transform('rank')

ranks['date'] = train.date

ranks[ranks.date == 'D000'].corr()

rank_cor = ranks.groupby('date').corr().reset_index()
ics = rank_cor[rank_cor.level_1=='fwd_returns']

ics.head()

avg_IR = ics.mean() / ics.std()

ranks

ics.loc[:,factors ].plot()
# better idea, calc x-section rank IC

# ===================
# run regression each month
import statsmodels.api as sm
regress_data = ranks.loc[ranks.date == 'D000',factors + ['fwd_returns']]
regress_data = regress_data.fillna(0)
y = regress_data.fwd_returns
X = regress_data.loc[:,factors]

model = sm.OLS(y,X).fit()
model.summary()

this_day = ranks.loc[ranks.date == 'D001',factors].fillna(0)

preds = model.predict(this_day)


# get predicted values

# use queue to cycle thru dates?


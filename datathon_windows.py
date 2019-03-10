# 3:43

import pandas as pd
import numpy as np
import os
import sys

print('hello')

os.getcwd()

raw_data = pd.read_csv('db_data.csv')

raw_data.tail()

raw_data.date.unique

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


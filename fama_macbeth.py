#


# https://randlow.github.io/posts/finance-economics/pandas-datareader-KF/
import statsmodels.api as sm
import statsmodels.formula.api as smf

import pandas_datareader.data as web  # module for reading datasets directly from the web
from pandas_datareader.famafrench import get_available_datasets

datasets = get_available_datasets()
print('No. of datasets:{0}'.format(len(datasets)))

df_5_factor = [dataset for dataset in datasets if '5' in dataset and 'Factor' in dataset]
print(df_5_factor)

ds_factors = web.DataReader(df_5_factor[0],'famafrench',
                    start='2006-01-31',
                            end='2015-12-31') # Taking [0] as extracting 1F-F-Research_Data_Factors_2x3')
print('\nKEYS\n{0}'.format(ds_factors.keys()))
print('DATASET DESCRIPTION \n {0}'.format(ds_factors['DESCR']))
ds_factors[0].head()

###########

data = pd.read_csv("stock_data_actual_dates.csv")

rets = data.loc[:,['stock','date','returns']]

rets['year_month'] = rets['date'].str.slice(0,7)

rets.info()

ff_5f = ds_factors[0]

ff_5f.index

################
ff_5f['year_month'] = ff_5f.index
ff_5f['year_month'] = ff_5f['year_month'].astype(str)

ff_5f.info()
rets.info()

ff_5f.RF.plot()

merged = pd.merge(rets, ff_5f, on=['year_month'], how = 'left')


df = merged.dropna()


############

# step 1
# run n regressions (one for each stock)
ff_factors = ['MKT','SMB','HML','RMW','CMA']

date_range = df.date.unique()
date_range.sort()

dt = date_range [0]

df_this_dt = df.query("date == @dt")

list_stocks = df.stock.unique()

stck = list_stocks[0]

stck

df.to_csv("ff_5_factors_and_returns.csv")

df = df.rename(columns = {'Mkt-RF':'MKT'})

df_one_stock = df.query("stock == @stck")

df_one_stock

X = df_one_stock.loc[:,ff_factors]
X['constant']=1

Y = df_one_stock['returns']

coefs = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))


results = smf.ols('returns ~ MKT + SMB + HML + RMW + CMA', data=df_one_stock).fit()
results.summary()
results.params


betas = df.groupby('stock').apply(run_ff_regression)

betas.iloc[:,1:]  # exclude intercept

# step 2)
# run T cross sectional regressions of returns ~ betas
#####################
T = gammas.shape[0]

df_betas = df.copy()

rets_2 = df_betas.loc[:,['stock','year_month','returns']]

df_step_2 = pd.merge(rets_2,
betas.iloc[:,1:].reset_index(), on=['stock'],how='left')

# time series of coefficients
gammas = df_step_2.groupby('year_month').apply(run_ff_regression)
gammas.plot()
# risk premiums
gammas.mean()

std_errs = gammas.std()/np.sqrt(T)

t_stats = gammas.mean()/std_errs
t_stats

##################


def run_ff_regression(df):
    """
    
    Parameters
    ----------
    df

    Returns
    -------

    """

    results = smf.ols('returns ~ MKT + SMB + HML + RMW + CMA', data=df).fit()
    return results.params











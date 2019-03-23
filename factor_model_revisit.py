# The Bean - Wed, March 20, 2019
##########################################
from scipy.stats import mstats

from ols_functions import (replace_with_dummies, zscore_but_ignore_binary_cols, \
    variance_inflation_factors, normalize)


X_raw = pd.read_csv('stock_data_renamed.csv')


X_raw.groupby('date').count()

#what to do with:
# missing returns? sector / mktcap ?

X_nona = X_raw.dropna()

X_nona.groupby('date').count()

X_nona['ln_mktcap'] = np.log(X_nona['mktcap'])

X_nona['mktcap'].hist(bins=100)

X_with_dummies = replace_with_dummies(df = X_nona, categorical_cols = 'sector',
                         leave_one_out=False)

cols_for_X = [x for x in X_with_dummies.columns if x not in ['returns','fwd_returns','mktcap']]

X = X_with_dummies.loc[:,cols_for_X ]


###############
# factor model object
##################
dts = pd.date_range(start='2005-12-31', end='2015-12-31', freq = 'BM')

X_raw = X_raw.reset_index()

{'D' + }

nums = [x for x in np.arange(0,120)]

dummy_dates = 'D'+pd.Series(nums ).astype(str).str.zfill(3)

date_mapper = pd.concat([dummy_dates, pd.Series(dts)],axis=1)
date_mapper.columns = ['dummy_date','date']

#date_mapper.date

X_new_dts = pd.merge(X_raw,
         date_mapper, left_on = 'date', right_on = 'dummy_date', how = 'left')
X_new_dts = X_new_dts.drop('date_x',axis=1).rename(columns = {'date_y':'date'})

X_new_dts.to_csv('stock_data_actual_dates.csv')

X_m = np.array(X)

            #import pdb; pdb.set_trace()

    #############
X_new_dts['size'] = np.log(X_new_dts['mktcap'])

model = factor_risk_model(X_new_dts)
model.calc_factor_ports_and_returns(list_dates=['2006-01-31','2006-02-28'],
    list_factors=['sector', 'momentum','quality','growth','vol','value','size'])

model.calc_factor_ports_and_returns(list_factors=['sector', 'momentum','quality','growth','vol','value','size'])
# doesnt work on 10-29-2010
# need to figure out hwy

X_new_dts.to_csv("")



X_new_dts[X_new_dts.date=='2010-10-29']

model.factor_returns

model.factor_portfolios.groupby(['date','factor']).sum().plot(kind='bar')


####################
X_new_dts.groupby('date').fwd_returns.count()


_df = X_new_dts[X_new_dts.date == '2006-01-31']

_df = _df.set_index('stock').loc[:,list_factors + ['fwd_returns']+['mktcap']]

_df = _df[_df.mktcap.notnull()]

_df = pd.get_dummies(_df,columns = ['sector'])

_df

col = _df.mktcap

col.unique()

set(_df['sector_4.0'].unique())=={0,1}

[is_binary(_df[x]) for x in _df.columns]







        ###############
# Create FMPs
##################
dt = 'D001'

X_one_day = X_with_dummies.query("date == 'D001'")

inv_vols = np.sqrt(X_one_day['mktcap'])

#inv_vols = X_one_day['mktcap']

DELTA = np.diag(inv_vols )

#D_df = pd.DataFrame(DELTA, index = X.index.get_level_values('stock'),
#             columns = X.index.get_level_values('stock'))

DELTA.shape

# does delta have to sum to 1?


# FMP = (X' * D * X)-1 * X' * D
#######################################

# winsorize
###################
df = X.copy()

df2 = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)

df2.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])

df.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])

###################

cols_no_dummies = cols_for_X[:6]

X = X_one_day.loc[:,cols_for_X  ]


X_ws = X.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)

X_z = zscore_but_ignore_binary_cols(X_ws )

X_z.describe()

_X = np.array(X_z)

#X_z.T.dot(D_df).dot(X_z).dot(D_df)
_X.shape

H = np.linalg.inv(_X.T.dot(DELTA).dot(_X)).dot(_X.T).dot(DELTA).T
# if i use weights, the portfolios are no longer dollar neutral why!?

H = np.linalg.inv(_X.T.dot(_X)).dot(_X.T).T

ols_get_coefs()

variance_inflation_factors_2(X_z, add_const= False).plot(kind='bar')

H.shape

pd.DataFrame(H).describe()
coef_mktcap = pd.DataFrame(H).sum(axis=0)
#coef_sqrt_mktcap = pd.DataFrame(H).sum(axis=0)

coefs_df = pd.DataFrame(H, index = X.index.get_level_values('stock'),
             columns = X.columns)
# nice its working!
coefs_df.to_csv('FMP_holdings.csv')

_x.shape

###########
# factor portfolios are dollar-neutral, but sector potfolios sum up to 1
# with weights being equal wt or porportional to risk


####################################

test_df = df.iloc[:10,:]

mom_normal = normalize(test_df.momentum )

rnks = test_df.momentum.rank()/(1+max(test_df.momentum.rank()))

norm.ppf(rnks )

norm.ppf([0.01,0.05, 0.50,0.95,.99])

##################################


def random_df():
    """
    
    Returns
    -------

    """


####################################


def variance_inflation_factors_2(exog_df, add_const = True):
    '''
    credit to : https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python

    Parameters
    ----------
    exog_df : dataframe, (nobs, k_vars)
        design matrix with all explanatory variables, as for example used in
        regression.

    Returns
    -------
    vif : Series
        variance inflation factors
    '''
    if add_const:
        exog_df = sm.add_constant(exog_df)
    vifs = pd.Series(
        [1 / (1. - sm.OLS(exog_df[col].values,
                          exog_df.loc[:, exog_df.columns != col].values).fit().rsquared)
         for col in exog_df],
        index=exog_df.columns,
        name='VIF'
    )
    return vifs







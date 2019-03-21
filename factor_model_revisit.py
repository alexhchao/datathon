# The Bean - Wed, March 20, 2019
##########################################
from scipy.stats import mstats

from ols_functions import (replace_with_dummies, zscore_but_ignore_binary_cols, \
    variance_inflation_factors)


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







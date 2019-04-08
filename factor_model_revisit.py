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

###########################
# START HERE - Sunday 4-8-2019
###########################

list_factors=['sector', 'momentum','quality','growth','vol','value','size']

df_new = pd.read_csv('stock_data_actual_dates.csv').iloc[:,1:]

#df_new = df_new.set_index('stock')

dt_list = list(df_new.date.unique()[48:])

##########
model = factor_risk_model(df_new)

model = factor_risk_model(df_new,
                          factor_portfolios = _factor_portfolios,
                          factor_returns = _factor_portfolio_returns,
                          specific_returns = _specific_returns,
                          all_factor_exposures = _all_factor_exposures)

# this takes a few min
model.calc_factor_ports_and_returns(list_dates= None,
    list_factors=['sector', 'momentum','quality','growth','vol','value','size'])

model.calculate_factor_cov_matrix(window = 60)
model.calculate_stock_covariance_matrix()
model.calculate_FMPs()

model.all_FMPs[dt]
model.all_FMPs_using_V[dt]

###############


cov_to_corr(model.all_stock_covariance_mat[dt])


model.all_specific_variances[dt].head()
############
# wait a sec FMP using V and DELTA yields the same hting? WTF ?
# WHY
# using diagonal of V doesnt yield the same thing
############

H_D = calculate_FMP(X = model.all_factor_exposures[dt],
              D = model.all_specific_variances[dt])

H_V = calculate_FMP(X = model.all_factor_exposures[dt],
              D = model.all_stock_covariance_mat[dt])

H_V - H_D

V = model.all_stock_covariance_mat[dt]
V_diag = np.diag(np.diagonal(model.all_stock_covariance_mat[dt]))
V_diag = pd.DataFrame(V_diag, index = V.index, columns = V.columns)

H_V_Diag = calculate_FMP(X = model.all_factor_exposures[dt],
              D = V_diag)



ident= np.identity(model.all_factor_exposures[dt].shape[0])
id = pd.DataFrame(ident, index = model.all_factor_exposures[dt].index,
        columns = model.all_factor_exposures[dt].index)

calculate_FMP(X = model.all_factor_exposures[dt],
              D = id)


D_inv = pd.DataFrame(np.linalg.inv(D),
                         index=D.index, columns=D.columns)

H = np.linalg.inv(X.T.dot(D_inv).dot(X)).dot(X.T.dot(D_inv))

H_1 = H.T
H_df = pd.DataFrame(H_1,
                    index=X.index, columns=X.columns)

V_inv = pd.DataFrame(np.linalg.inv(V),
                         index=V.index, columns=V.columns)

H_V = np.linalg.inv(X.T.dot(V_inv).dot(X)).dot(X.T.dot(V_inv))

D
V

def calculate_FMP(X, D):
    """
    
    Parameters
    ----------
    H
    W

    Returns
    -------

    """

    D_inv = pd.DataFrame(np.linalg.inv(D),
                         index=D.index, columns=D.columns)

    H = np.linalg.inv(X.T.dot(D_inv).dot(X)).dot(X.T.dot(D_inv))
    H_1 = H.T
    H_df = pd.DataFrame(H_1,
                        index=X.index, columns=X.columns)
    return H_df


############
# lets run some attributions!
# NEXT STEP:
# need to form new FMPs! using the _DELTA from risk model!
###########
# H' = (X * D * X')-1 * X' D
# H * v * w = 0
# P = H * B + w
dt = '2015-12-31'

model.all_stock_covariance_mat
model.specific_returns

D = model.all_specific_variances[dt]

D_inv = pd.DataFrame(np.linalg.inv(D), index= D.index, columns = D.columns)

X = model.all_factor_exposures[dt]

_all_factor_exposures = model.all_factor_exposures

#_X = np.as_array(X)
H = np.linalg.inv(X.T.dot(D_inv).dot(X)).dot(X.T.dot(D_inv))
H_1 = H.T
H_1.shape

H_df = pd.DataFrame(H_1,
                 index=X.index, columns = X.columns)

# this should yield the identity matrix
H_X = H_df.T.dot(X)

H_X ==np.identity(H_X .shape[0])


##################
# now pick one factor and run attribution on it

_F = factor_cov.loc[list_factors ,list_factors]
list_factors = list(_F.index)

factor_name = 'momentum'
factor_port = model.all_FMPs[dt][factor_name]
list_factors_in_S = [x for x in list_factors if x != factor_name]
other_ports = model.all_FMPs[dt][list_factors_in_S ]

factor_exp = factorAttribution(V = np.array(model.all_stock_covariance_mat[dt]),
                               F = _F.loc[list_factors_in_S, list_factors_in_S],
                     h = np.array(factor_port) ,
                     S = np.array(other_ports) ,
                     R = np.array(fwd_rets),
                   list_factors = list_factors_in_S)

factor_exp

# hmmm momentum FMP isnt orthogonal to other factors..
# what if we throw in momentum into the right hand side?
_V = model.all_stock_covariance_mat[dt]

_V_inv = pd.DataFrame(np.linalg.inv(_V), index = _V.index, columns = _V.columns)

np.linalg.inv(X.T.dot(D_inv).dot(X)).dot(X.T.dot(D_inv))

H_2 =pd.DataFrame(np.linalg.inv(D), index=D.index, columns=D.columns)

factor_name = 'momentum'
factor_port = model.all_FMPs[dt][factor_name]
list_factors_in_S = [list_factors]
other_ports = model.all_FMPs[dt][list_factors]

factor_exp = factorAttribution(V = np.array(model.all_stock_covariance_mat[dt]),
                               F = _F.loc[list_factors, list_factors],
                     h = np.array(factor_port) ,
                     S = np.array(other_ports) ,
                     R = np.array(fwd_rets),
                   list_factors = list_factors)
factor_exp
H = other_ports

H.dot(V).dot()

##################

dt

factor_port = model.factor_portfolios.query("date==@dt").query(" factor=='momentum' ").set_index('stock').weight
other_ports = model.factor_portfolios.query("date==@dt").query(" factor!='momentum' ").set_index(['stock','factor']).weight.unstack()

factor_this_dt = model.factor_portfolios.query("date==@dt")

factor_this_dt[factor_this_dt.factor.isin(list_factors)].set_index(['stock','factor']).weight.unstack()

list_stocks = factor_port.index
fwd_rets = model.df.query("date == @dt").reset_index()[model.df.query(
    "date == @dt").reset_index().stock.isin(list_stocks)].set_index('stock').fwd_returns*0.01

list_factors = other_ports.columns

factor_cov = model.all_factor_covariance_mat[dt]


factor_exp = factorAttribution(V = np.array(model.all_stock_covariance_mat[dt]),
                               F = np.array(factor_cov.loc[list_factors ,list_factors ]),
                     h = np.array(factor_port) ,
                     S = np.array(other_ports) ,
                     R = np.array(fwd_rets),
                               list_factors = list_factors)
factor_exp

factor_exp.risk_contrib_from_factors_out.sum(axis=0)[0]

[x for x in factor_exp.return_contrib_from_factors]

# factor returns
factor_rets = factor_exp.S.T.dot(R)

factor_exp.port_factor_exposure.dot(factor_rets )

factor_exp.S.dot(factor_exp.port_factor_exposure)

factor_exp.factor_vol


#############
# save down factor and spec returns!

_factor_portfolios = model.factor_portfolios.copy()
_factor_portfolio_returns = model.factor_returns.copy()
#model.factor_wealth.iloc[:,:6].plot()
_specific_returns = model.specific_returns.copy()

##############


cov_to_corr()
np.diagonal(model.all_specific_variances['2015-12-31'])*1000

##########
# calc nxn risk model! yay! finally

V = X * F * X.T + DELTA

dt = '2015-12-31'

X = model.all_factor_exposures[dt]
_F = model.all_covariance_mat[dt]

_DELTA = model.all_specific_variances[dt]

_DELTA.shape

X.dot(_F).dot(X.T) + _DELTA

###########

model.factor_returns
model.n


dt = '2006-03-31'

model.specific_returns.loc[dt].dropna()
model.all_factor_exposures[dt]

#model.df.describe()

model.factor_portfolios.groupby(['date','factor']).weight.sum()

model.calculate_factor_cov_matrix(window=36)

model.df

u = model.specific_returns

u.loc[:,['S726','S727','S728']].iloc[:12,:].to_clipboard()

r = np.array([.02, -.02, .55, -.55])

w = get_exp_weights(window = 4,half_life = 2)

get_exp_weights(window = 16,half_life = 8)

[(0.5)**((12-t)/6) for t in range(1,13)]



model_2 = factor_risk_model(df = df_new)
model_2.calc_factor
    ,
                           # factor_portfolios=_factor_portfolios,
                           # factor_returns=_factor_portfolio_returns,
                           # specific_returns=_specific_returns
                            )
model_2.calculate_factor_cov_matrix()

model_2.all_specific_variances

#######################

r.T.dot(r)/4
r.T.dot(np.diag(w)).dot(r)


calc_exp_wt_cov_mat(u.iloc[:12,:].T, w)

model.all_corr_mat

model.all_cov_mat

model.factor_returns
# specific returns
u = r - X*f

# V = XFX' + delta
# r = X * f + u
# STILL TO DO : calc specific variance!



#model.calc_factor_ports_and_returns(list_factors=['sector', 'momentum','quality','growth','vol','value','size'])
# doesnt work on 10-29-2010
# need to figure out hwy
# issue with taking max of series with NA!

model.factor_portfolios.groupby(['date','factor']).weight.sum().unstack().describe()

np.cumprod(1+model.factor_returns).to_csv('factor_wealth.csv')

model.factor_returns.to_csv('FMP_returns.csv')

################################
# calc factor cov matrix!
#######################################
# covariance   V = F*F' / (T-1)
# weighted cov V   =   F    W   F'   / (T-1)
#              kxk     kxT  TxT Txk
T = model.factor_returns.T.shape[1]
T = total months



w = get_exp_weights(window = window, half_life = 6)


W.columns = F.columns
W.index = F.columns

cov_real = model.factor_returns.cov()

T

np.diag([1,2,3])

################
# calc factor cov matrix!
#######################################
window = 12
w = get_exp_weights(window = window, half_life = window/2)

all_cov_mat = {}
all_corr_mat = {}

i=0
for i in range(0,T-window+1):
    print(i)
    F = model.factor_returns[i:i+window].T
    last_dt = F.columns[-1]
    _V = calc_exp_wt_cov_mat(F, w)
    _C = cov_to_corr(_V)

    all_cov_mat[last_dt] = _V
    all_corr_mat[last_dt] = _C

all_corr_mat



############


corr_to_cov(C = cov_to_corr(_V), D = np.diag(_V) )

__v = D.dot(_C).dot(D)

pd.DataFrame(__v )







N = F.shape[1]

F.T.cov()

F.mean(axis=1)
_f = F.apply(lambda x: x-x.mean())

#_f.T.cov()

cov_mat = F.dot(F.T)/(N-1)
cov_mat_w = F.dot(W).dot(F.T)/(N-1)
cov_mat_w

#######
# convert cov to corr
D = np.diag(np.sqrt(np.diagonal(cov_mat_w)))

cov_to_corr(V)

corr = np.linalg.inv(D).dot(cov_mat_w).dot(np.linalg.inv(D))
pd.DataFrame(corr )

######

F.mean(axis=1)

cov_real -F.dot(F.T)/(N-1)

F.dot(F.T)/(N-1)

F.dot()

#######################################
# exp wets
# w_t = (2/w_sum)~(-(T-t)/lambda)

lamb = 60
T = 120

[()t for t in np.arange(1,T+1)]

##############
# covariance matrix decomposition
#################################
# V = D * C * D'
# C = correlation matrix
# D = diagonal of volatilities
D = np.sqrt(np.diag(V))



#######################################

X_new_dts.to_csv("")


df = df_new.copy()

df.index.names[0]==None

df = df.set_index(['date','stock'])

list(df.index.names)==['date','stock']



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







# 8-14-2019
###########
import pandas as pd
import numpy as np

data = pd.read_csv("./secor/raw_data.csv")
data.columns
data.columns = ['date', 'FX_MXN', 'FX_SGD', 'FX_ZAR', 'FX_INR', 'FX_BRL', 'FX_TRY']
data.date = pd.to_datetime(data.date)

######
#1 ) write function that computes simple covariance matrix
data

X = data
X = X.set_index('date')

N = X.shape[0]

X.T.dot(X)*(1/N-1)

Xa = X.query("date >= '2002-01-01' ")

Xa.T.dot(Xa)

# expanding window
start = min_obs = 260
dt_rng = list(Xa.index.unique())

max_obs = 2600


def calc_cov(X):
    return X.T.dot(X)*(1/(X.shape[0]-1))


all_cov = {}

for dt in dt_rng[start:]:

    dt_str = dt.strftime("%Y-%m-%d")

    _X = Xa.loc[dt_rng[0]:dt]
    if _X.shape[0] > max_obs:
        _X = _X.iloc[-max_obs:]
    V = calc_cov(_X)
    #_X.T.dot(_X)*(1/N)
    print(dt)
    all_cov[dt_str] = V
#    cov_to_corr(V)

 all_cov['2015-12-31']


all_cov = covariance(returns = Xa, freq= 260)

#############

def covariance(returns,
               min_obs = 260,
               max_obs = 2600,
               freq = 1):
    """
    
    Parameters
    ----------
    returns
    min_obs
    max_obs
    freq

    Returns
    -------
    NxNxT cov matrix by time
    """

    dt_rng = list(returns.index.unique())

    all_cov = {}
    start = min_obs-1
    for dt in dt_rng[start:]:

        dt_str = dt.strftime("%Y-%m-%d")

        _X = returns.loc[dt_rng[0]:dt]
        if _X.shape[0] > max_obs:
            _X = _X.iloc[-max_obs:]
        V = calc_cov(_X)
        # _X.T.dot(_X)*(1/N)
        print(dt)
        all_cov[dt_str] = V*np.sqrt(freq)
        #    cov_to_corr(V)

    return all_cov

# 2) decompose cov matrix into:
#    correlations
#    vol


def decompose_cov_into_corr_vol(V):
    """
    
    Parameters
    ----------
    V

    Returns
    -------

    """

    D = np.diag(np.sqrt(np.diagonal(V)))
    C = np.linalg.inv(D).dot(V).dot(np.linalg.inv(D))
    return C, D



def vol_floor(C,D):
    """
    outputs where vol is >= floor 
    if not ?
    
    Parameters
    ----------
    C
    D

    Returns
    -------

    """
    D

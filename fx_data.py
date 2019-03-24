# Argo - Sun, March 24, 2019
##########################################
fx_data = pd.read_csv('fx_data.csv')

fx_data = fx_data.rename(columns = {'Unnamed: 0':'date'})

fx_data.date = pd.to_datetime(fx_data.date)


fx_data =  fx_data.set_index('date')


#############
# calc cov matrix

cov_real = model.factor_returns.cov()

F = model.factor_returns.T
N = F.shape[1]

F.mean(axis=1)

cov_real -F.dot(F.T)/(N-1)

F.dot(F.T)/(N-1)



calc_cov_mat(F = fx_data.T)


def numpy_ewm_alpha_v2(a, alpha, windowSize):
    """
    https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    
    Parameters
    ----------
    a
    alpha
    windowSize

    Returns
    -------

    """
    wghts = (1-alpha)**np.arange(windowSize)
    wghts /= wghts.sum()
    out = np.convolve(a,wghts)
    out[:windowSize-1] = np.nan
    return out[:a.size]



def calc_cov_mat(F,
                 assume_mean_zero = True,
                 W = None
                 ):
    """
    calculate covariance matrix
    
    Parameters
    ----------
    F - matrix or pd.DataFrame (kxT), T = total time periods
    assume_mean_zero - default = True
    W - weights (e.g. exponential weights

    Returns
    -------

    """

    N = F.shape[1]

    if W is None and assume_mean_zero:
        return F.dot(F.T)/(N-1)






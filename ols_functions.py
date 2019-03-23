# June 30, 2018
# saturday
# argo

###################
# function to use with statsmodels api

import pandas as pd
import numpy as np

pd.options.display.max_rows = 10

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy.stats import zscore
from scipy.stats import norm


def check_if_matrix_has_nans(m):
    return np.any(np.isnan(m ))


def normalize(x):
    """
    percentile rank than inverse normal dist
    
    Parameters
    ----------
    x

    Returns
    -------

    """
    ranks = x.rank()
    _x = ranks/(1+max(ranks))
    return pd.Series(norm.ppf(_x))


def zscore_but_ignore_binary_cols(df,
                                  func_to_apply=normalize):
    """


    Parameters
    ----------
    df

    Returns
    -------

    """
    _df = df.copy()

    numeric_cols = _df.columns[[is_not_binary(_df[x]) for x in _df.columns]]
    binary_cols = _df.columns[[is_binary(_df[x]) for x in _df.columns]]

    _df_numeric = _df.loc[:, numeric_cols]
    _df_binary = _df.loc[:, binary_cols]

    _df_z = _df_numeric.apply(func_to_apply, axis=0).replace(np.NaN, 0.00)
    _df_z.columns = _df_numeric.columns
    _df_z.index = _df_binary.index
    print(_df_z.describe())

    out = pd.concat([_df_z, _df_binary], axis=1)

    if out.shape != _df.shape:
        print(" shapes not correct! something is wrong")
        import pdb; pdb.set_trace()

    return out


def is_binary(col):
    """


    Parameters
    ----------
    col

    Returns
    -------

    """
    if not isinstance(col, pd.Series):
        raise ValueError("column is not a series! please try again")

    return set(col.unique()) == {0, 1}


def is_not_binary(col):
    """


    Parameters
    ----------
    col

    Returns
    -------

    """
    if not isinstance(col, pd.Series):
        raise ValueError("column is not a series! please try again")

    return not set(col.unique()) == {0, 1}


def ols_get_coefs(X, y, w=None):
    """
    (X'X)-1 * (X.T * y)

    Parameters
    ----------
    X
    y

    Returns
    -------

    """
    if w is not None:
        return np.linalg.inv(X.T.dot(w).dot(X)).dot(X.T.dot(w).dot(y))
    else:
        return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))


# calc VIF
def variance_inflation_factors(exog_df, add_const = True):
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


def plot_pred_vs_actual(pred, actual):
    """

    Parameters
    ----------
    pred
    actual

    Returns
    -------

    """
    plt.scatter(pred, actual, label='medv')
    plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
    plt.xlabel('pred')
    plt.ylabel('actual')


def zscore_but_ignore_binary_cols(df,
                                  func_to_apply=zscore):
    """
    returns column zscores but keeps binary variables intact

    Parameters
    ----------
    df

    Returns
    -------

    """
    _df = df.copy()

    list_bin_cols = [c for c in _df.columns if is_binary(_df[c])]

    df_numeric = _df.loc[:, [c for c in _df.columns if c not in list_bin_cols]]
    df_binaries = _df.loc[:, list_bin_cols]
    return pd.concat([df_numeric.apply(func_to_apply, axis=0),
                      df_binaries], axis=1)


def is_binary(_series):
    """
    checks to see if a series is binary
    Parameters
    ----------
    _series

    Returns
    -------

    """
    return list(set(_series)) == [0, 1]


def replace_with_dummies(df, categorical_cols,
                         leave_one_out=True):
    """

    Parameters
    ----------
    df
    categorical_cols
    leave_one_out = for each categorical col, leave one out

    Returns
    -------

    """
    _df = df.copy()

    if leave_one_out:

        _all_dums = []
        for c in categorical_cols:
            _d_df = pd.get_dummies(_df[c]).iloc[:, 1:]
            _d_df.columns = c + '_' + _d_df.columns
            _all_dums.append(_d_df)

        _dummies = pd.concat(_all_dums, axis=1)
        # _dummies = pd.concat([pd.get_dummies(
        # df[c]).iloc[:, 1:] for c in categorical_cols], axis=1)
    else:
        _dummies = pd.get_dummies(_df.loc[:, categorical_cols])

    return pd.concat([_df.drop(categorical_cols, axis=1), _dummies], axis=1)


def plot_fitted_vs_resids(model):
    """
    plots fitted vs resids plot

    Parameters
    ----------
    ols_model - model from statsmodels

    Returns
    -------
    df
    """
    if not isinstance(model, sm.regression.linear_model.RegressionResultsWrapper):
        raise ValueError("model is not a statsmodels model!")

    ax = sns.regplot(x=model.fittedvalues, y=model.resid, lowess=True)
    ax.set(xlabel='Fitted values', ylabel='Residuals')
    plt.show()


def mse(model):
    """

    Parameters
    ----------
    model

    Returns
    -------
    MSE - float
    """
    if not isinstance(model, sm.regression.linear_model.RegressionResultsWrapper):
        raise ValueError("model is not a statsmodels model!")

    return (model.resid ** 2).mean()



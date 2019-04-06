# utility functions


def calc_exp_wt_cov_mat(F, w):
    """

    Parameters
    ----------
    F - factor returns (k x T)
    w = vector of exp weights
    Returns
    -------

    """
    W = np.diag(w)
    W = pd.DataFrame(W)

    W.columns = F.columns
    W.index = F.columns

    return F.dot(W).dot(F.T) / (len(w) - 1)

def corr_to_cov(C, D):
    """

    Parameters
    ----------
    C
    D

    Returns
    -------

    """
    D = np.diag(D)
    if isinstance(C, pd.DataFrame):
        V = D.dot(C).dot(D)
        #import pdb; pdb.set_trace()
        V = pd.DataFrame(V)

        V.columns = C.columns
        V.index = C.index
        return V
    else:
        return D.dot(C).dot(D)


def cov_to_corr(V):
    """

    Parameters
    ----------
    V

    Returns
    -------

    """
    D = np.diag(np.sqrt(np.diagonal(V)))

    if isinstance(V, pd.DataFrame):
        C = np.linalg.inv(D).dot(V).dot(np.linalg.inv(D))
        C = pd.DataFrame(C)
        C.columns = V.columns
        C.index = V.index
        return C
    else:
        return np.linalg.inv(D).dot(V).dot(np.linalg.inv(D))




def get_exp_weights(window,
                    half_life,
                    order = 'sorted'):
    """
    
    Parameters
    ----------
    window
    half_life
    order

    Returns
    -------

    """
    exp_wts = 0.5**(np.arange(0, window)/half_life)
    exp_wts = exp_wts / sum(exp_wts)
    if order == 'sorted':
        return np.sort(exp_wts)
    else:
        return exp_wts

def smart_set_index(df, index):
    """
    sets index if not already set yet

    Parameters
    ----------
    df
    index

    Returns
    -------

    """
    # no index set, then set index
    if df.index.names[0] is None:
        return df.set_index(index)
    elif list(df.index.names) == index:
        return df
    else:
        return df.reset_index().set_index(index)




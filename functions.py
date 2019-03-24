# utility functions


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




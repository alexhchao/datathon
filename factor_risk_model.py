#from ols_functions import (zscore_but_ignore_binary_cols,
#                           normalize, check_if_matrix_has_nans,
#                           is_binary, is_not_binary)

class factor_risk_model(object):
    """
    factor risk model
    """

    def __init__(self,
                 df,
                 sector_col='sector',
                 mktcap_col='mktcap',
                 fwd_ret_col='fwd_returns',
                 stock_col='stock',
                 date_col='date'):

        self.sector_col = sector_col
        self.mktcap_col = mktcap_col
        self.fwd_ret_col = fwd_ret_col
        self.stock_col = stock_col
        self.date_col = date_col
        self.df = smart_set_index(df, index = [date_col, stock_col])
        self.factor_portfolios = None
        self.factor_returns = None

    @property
    def dates(self):
        return list(self.df.index.get_level_values(self.date_col).unique())

    def calc_factor_ports_and_returns(self,
                                      list_dates=None,
                                      list_factors=None,
                                      func_to_apply=normalize):
        """

        Parameters
        ----------
        list_factors

        Returns
        -------

        """
        _df_all = self.df.copy()

        if list_dates is None:
            list_dates = self.dates

        if list_factors is None:
            list_factors = [self.sector_col]

        if self.sector_col in list_factors:
            print("filtering out when {} is null".format(self.sector_col))
            _df_all = _df_all[_df_all[self.sector_col].notnull()]

        print("filtering out when {} is null".format(self.mktcap_col))
        _df_all = _df_all[_df_all[self.mktcap_col].notnull()]

        all_factor_ports = []
        all_factor_rets = []

        for dt in list_dates:
            print(dt)

            #import pdb; pdb.set_trace()

            #_df = _df_all[_df_all[self.date_col] == dt]
            _df = _df_all[_df_all.index.get_level_values(
                self.date_col) == dt].reset_index(level=0)


            #_df = _df.loc[:, list_factors + [self.stock_col] + [self.mktcap_col] + [self.fwd_ret_col]]
            _df = _df.loc[:, list_factors + [self.mktcap_col] + [self.fwd_ret_col]]

            #_df = _df.set_index(self.stock_col)
            # get returns vector
            r_df = _df.loc[:, [self.fwd_ret_col]] * 0.01

            # remove outliers
            r_df[r_df > 10] = np.NaN
            r_df[r_df < -1] = np.NaN

            r_df = r_df.replace(np.NaN, 0.000)

            R = np.array(r_df)

            # list stock names
            all_stock_names = list(_df.index)

            W = np.diag(np.sqrt(_df[self.mktcap_col]))

            if self.sector_col in list_factors:
                _df = pd.get_dummies(_df, columns=[self.sector_col])

            # take out returns and mktcap
            _df = _df.loc[:, [c for c in _df.columns if c not in [self.fwd_ret_col,
                                                                  self.mktcap_col]]]

            all_factor_names = list(_df.columns)

            X = zscore_but_ignore_binary_cols(_df)
            #import pdb; pdb.set_trace()

            if any([x for x in X.count(axis=0).values if x == 0]):
                print('ERROR! one or more factors are all NA')
                import pdb;pdb.set_trace()

            X = np.array(X)

            try:
                if check_if_matrix_has_nans(X):
                    raise ValueError("X has NA values!")
                if check_if_matrix_has_nans(R):
                    raise ValueError("R has NA values!")
                if check_if_matrix_has_nans(W):
                    raise ValueError("W has NA values!")
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()


            f_df, rets_df = self.get_factor_ports_and_returns(X, W, R,
                                                              all_factor_names=all_factor_names,
                                                              all_stock_names=all_stock_names,
                                                              stock_col=self.stock_col,
                                                              dt=dt,
                                                              stack=True)

            all_factor_ports.append(f_df)
            all_factor_rets.append(rets_df)

        self.factor_portfolios = pd.concat(all_factor_ports, axis=0)
        self.factor_returns = pd.concat(all_factor_rets, axis=1).T

        print(self.factor_portfolios.head())
        print(self.factor_portfolios.head())

        print("factor portfolios saved under factor_portfolios!")
        print("factor returns saved under factor_returns!")


    @staticmethod
    def get_factor_ports_and_returns(X, W, R,
                                     all_factor_names,
                                     all_stock_names,
                                     dt,
                                     stock_col='stock',
                                     date_col='date',
                                     stack=True):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
        try:
            f = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).T
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
        rets = f.T.dot(R)

        # import pdb;pdb.set_trace()

        rets_df = pd.DataFrame(rets, index=all_factor_names, columns=[dt])

        f_df = pd.DataFrame(f, index=all_stock_names, columns=all_factor_names)

        if stack:
            f_df = f_df.stack().reset_index()
            f_df.columns = [stock_col, 'factor', 'weight']
            f_df[date_col] = dt
            return f_df, rets_df
        else:
            return f_df, rets_df

    def factor_portfolio_stats(self):
        if self.factor_portfolios is not None:
            return self.factor_portfolios.groupby(
            [self.date_col,'factor']).weight.sum().unstack().describe()
        else:
            return None

    def factor_wealth(self):
        if self.factor_returns is not None:
            return np.cumprod(1+self.factor_returns)
        else:
            return None

    def __repr__(self):
        print("""
        risk_model_factor object
        ------------------------
        {}
        
        """.format(self.df))


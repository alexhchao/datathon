import pandas as pd
import numpy as np

from ols_functions import (normalize, zscore_but_ignore_binary_cols,check_if_matrix_has_nans,
                           is_binary, is_not_binary)
from functions import (smart_set_index, get_exp_weights, cov_to_corr)

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
                 date_col='date',
                 factor_portfolios = None,
                 factor_returns = None,
                 specific_returns = None,
                 all_factor_exposures = None):

        self.sector_col = sector_col
        self.mktcap_col = mktcap_col
        self.fwd_ret_col = fwd_ret_col
        self.stock_col = stock_col
        self.date_col = date_col
        self.df = smart_set_index(df, index = [date_col, stock_col])
        self.factor_portfolios = factor_portfolios
        self.factor_returns = factor_returns
        self.specific_returns = specific_returns
        self.all_factor_exposures = all_factor_exposures
        self.all_factor_covariance_mat = {}
        self.all_factor_correlation_mat = {}
        self.all_specific_variances = {}
        self.valid_dates_for_covariance_matrix = []
        self.all_FMPs = {}

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

        print("filtering out when {} is null".format(self.fwd_ret_col))
        _df_all = _df_all[_df_all[self.fwd_ret_col].notnull()]

        all_factor_ports = []
        all_factor_rets = []
        all_specific_rets = []
        self.all_factor_exposures = {}
        self.n = {}
        self.fwd_rets = {}

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
            r_df[r_df > 10] = 10
            r_df[r_df < -1] = -.99

            self.fwd_rets[dt] = r_df

            #import pdb;pdb.set_trace()

            ##########
            # exclude all stocks whose return is NA
            #r_df = r_df.dropna()
            #r_df = r_df.replace(np.NaN, 0.000) # might want to exlcude when returns is NULL

            self.n[dt] = _df.shape[0]

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

            #import pdb;pdb.set_trace()
            self.all_factor_exposures[dt] = X

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


            f_df, rets_df, spec_returns_df = self.get_factor_ports_and_returns(X, W, R,
                                                              all_factor_names=all_factor_names,
                                                              all_stock_names=all_stock_names,
                                                              stock_col=self.stock_col,
                                                              dt=dt,
                                                              stack=True)

            all_factor_ports.append(f_df)
            all_factor_rets.append(rets_df)
            all_specific_rets.append(spec_returns_df)

            #import pdb; pdb.set_trace()

        self.factor_portfolios = pd.concat(all_factor_ports, axis=0)
        # need to shift returns by one unit ? To shift or not to shift, that is the question
        self.factor_returns = pd.concat(all_factor_rets, axis=1).T
        self.specific_returns = pd.concat(all_specific_rets, axis=1).T
        #self.factor_exposures = pd.concat(all_factor_exposures, axis=0)
        self.n = pd.DataFrame(self.n, index=['n_stocks']).T

        #import pdb; pdb.set_trace()
        print(self.factor_portfolios.head())
        print(self.factor_returns.head())
        print(self.specific_returns.head())

        print("factor portfolios saved under factor_portfolios!")
        print("factor returns (f) saved under factor_returns!")
        print("specific returns (u) saved under specific_returns!")
        print("all factor exposures (X) saved under all_factor_exposures!")

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

        #f_df = factor portfolio holdings
        f_df = pd.DataFrame(f, index=all_stock_names, columns=all_factor_names)

        # calculate specific returns here
        # r = X f + u
        # u = r - X * f
        # import pdb; pdb.set_trace()
        spec_returns = R - X.dot(rets_df)
        spec_returns_df = pd.DataFrame(spec_returns,
                                       index=all_stock_names,
                                       columns=[dt])

        if stack:
            f_df = f_df.stack().reset_index()
            f_df.columns = [stock_col, 'factor', 'weight']
            f_df[date_col] = dt

        return f_df, rets_df, spec_returns_df

    def calculate_factor_cov_matrix(self,
                                    window = 12,
                                    use_exp_weights = False):
        """
        
        Parameters
        ----------
        window

        Returns
        -------

        """
        w = get_exp_weights(window=window, half_life=window / 2)

        self.all_factor_covariance_mat = {}
        self.all_factor_correlation_mat = {}
        self.all_specific_variances = {}
        self.valid_dates_for_covariance_matrix = []

        i = 0
        T = self.factor_returns.shape[0]

        for i in range(0, T - window + 1):

            F = self.factor_returns[i:i + window]
            _u = self.specific_returns[i:i + window]

            last_dt = F.index[-1]
            print(last_dt)
            self.valid_dates_for_covariance_matrix.append(last_dt)

            # filter down _u into valid stocks only (no NA)
            valid_stocks = list(self.all_factor_exposures[last_dt].index)
            _u_valid = _u.loc[:, valid_stocks]
            if use_exp_weights:
                # NOT IMPLEMENTED YET!
                raise NotImplementedError("Not implemented exp weighted cov yet! ")
                #_V = calc_exp_wt_cov_mat(F, w)
            else:
                _V = F.cov()
                spec_variances = _u_valid.var(axis=0)
                #import pdb; pdb.set_trace()
                # set all missing variances to the median (e.g. if stock only has return history of 1 date)
                spec_variances = spec_variances.replace(np.NaN, spec_variances.median() )
                _DELTA = np.diag(spec_variances)

                if check_if_matrix_has_nans(_DELTA):
                    import pdb;pdb.set_trace()
                    print("_DELTA on {} has NA!".format(last_dt))

                _DELTA = pd.DataFrame(_DELTA, index=valid_stocks, columns = valid_stocks)

            _C = cov_to_corr(_V)
            #import pdb;pdb.set_trace()
            self.all_factor_covariance_mat[last_dt] = _V
            self.all_factor_correlation_mat[last_dt] = _C
            self.all_specific_variances[last_dt] = _DELTA

        print('all factor cov matrices (F) saved under all_factor_covariance_mat')
        print('all factor corr matrices saved under all_factor_correlation_mat')
        print('all specific variances (DELTA) saved under all_specific_variances')

    def calculate_stock_covariance_matrix(self):
        """
        
        Returns
        -------

        """
        self.all_stock_covariance_mat = {}

        for dt in self.valid_dates_for_covariance_matrix:
            print(dt)

            # V = X * F * X.T + DELTA
            X = self.all_factor_exposures[dt]
            _F = self.all_factor_covariance_mat[dt]
            _DELTA = self.all_specific_variances[dt]

            self.all_stock_covariance_mat[dt] = X.dot(_F).dot(X.T) + _DELTA

        print('all stock by stock cov matrices (V) saved under all_stock_covariance_mat')

    def calculate_FMPs(self,
                       #V_or_D = 'D',
                       list_factors = None):
        """
        
        Parameters
        ----------
        V_or_D - use V (cov matrix) or D (spec variance)
        list_factors

        Returns
        -------

        """

        self.all_FMPs = {}
        self.all_FMPs_using_V = {}

        for dt in self.valid_dates_for_covariance_matrix:
            print(dt)

            if list_factors is None:
                X = self.all_factor_exposures[dt]
            else:
                X = self.all_factor_exposures[dt].loc[:,list_factors]
            #if V_or_D == 'D':
            D = self.all_specific_variances[dt]
            #print(D.head())

            #D_inv = pd.DataFrame(np.linalg.inv(D), index=D.index, columns=D.columns)
        #elif V_or_D == 'V':
            V = self.all_stock_covariance_mat[dt]
            #print(D.head())
            #V_inv = pd.DataFrame(np.linalg.inv(V), index=V.index, columns=V.columns)

            X = self.all_factor_exposures[dt]

            H_df = self.calculate_FMP(X = X,
                                      D = D)
            H_df_using_V = self.calculate_FMP(X = X,
                                              D = V)

            # _X = np.as_array(X)
            #H = np.linalg.inv(X.T.dot(D_inv).dot(X)).dot(X.T.dot(D_inv))
            #H_1 = H.T

            #H_df = pd.DataFrame(H_1,
            #                    index=X.index, columns=X.columns)

            #H_V = np.linalg.inv(X.T.dot(V_inv).dot(X)).dot(X.T.dot(V_inv))
            #H_V_1 = H_V.T

            #H_df_using_V = pd.DataFrame(H_V_1,
            #                    index=X.index, columns=X.columns)
            self.all_FMPs[dt] = H_df
            self.all_FMPs_using_V[dt] = H_df_using_V

        print('all FMPs (H) saved under all_FMPs')
        print('all FMPs (H) (using V instead of D) saved under all_FMPs_using_V')

    @staticmethod
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



    @property
    def get_factor_portfolio_holdings(self,
                                      date,
                                      factors):
        """
        # these are the factor portfolios from our risk model
        H' = H' = (X * W * X')-1 * X' W
        
        where
        W = sqrt(mktcap)
        
        Parameters
        ----------
        date
        factor

        Returns
        -------

        """
        if isinstance(factors, str):
            return self.factor_portfolios.query("date==@date").query(
                       " factor==@factors ").set_index('stock').weight
        elif isinstance(factors, list):
            factor_this_dt = self.factor_portfolios.query("date==@dt")
            return factor_this_dt[factor_this_dt.factor.isin(
                factors)].set_index(['stock','factor']).weight.unstack()

    @property
    def factor_portfolio_stats(self):
        if self.factor_portfolios is not None:
            return self.factor_portfolios.groupby(
            [self.date_col,'factor']).weight.sum().unstack().describe()
        else:
            return None

    @property
    def factor_wealth(self):
        if self.factor_returns is not None:
            return np.cumprod(1+self.factor_returns)
        else:
            return None

    def __repr__(self):
        return("""
        risk_model_factor object
        ------------------------
        {}
        
        """.format(self.df))


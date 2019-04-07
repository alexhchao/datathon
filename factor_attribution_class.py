import numpy as np
import pandas as pd

class factorAttribution(object):
    """
    factor risk model
    """

    def __init__(self,
                 V,
                 F,
                 h,
                 S,
                 R):
        """
        
        Parameters
        ----------
        V - stock covariance matrix (n x n)
        F - factor covariance matrix (k x k)
        h - holdings (n x 1)
        S - source or factor portfolios (n x k), k = num factors
        R - future returns (n x 1)
        
        """
        if V.shape[0] != V.shape[1]:
            raise ValueError(" Cov matrix V is not square!")

        self.V = V # n x n
        self.F = F # k x k
        self.h = h # n x 1
        self.S = S # n x k
        self.R = R # n x 1

        self.n = h.shape[0]
        self.k = S.shape[1]
        if self.check_all_dims_align() == False:
            print(" V={}, h={}, S={}, R={}".format(V.shape,
                                                   h.shape,
                                                   S.shape,
                                                   R.shape))
            raise ValueError(" Dimensions not aligned! please check again")

        # calc factor risk / return contributions

        self.port_var = self.h.T.dot(self.V).dot(self.h)  # 7.86%
        self.port_vol = np.sqrt(self.port_var)  # 28.02%

        # A) Factor exposure
        # B = (S' * V * S)-1 * S' * V * h
        B = np.linalg.inv(self.S.T.dot(self.V).dot(self.S)).dot(self.S.T.dot(self.V).dot(self.h))

        self.port_factor_exposure = B
        #self.port_factor_exposure = self.h.dot(self.S)

        # B) Vol Adj Exposure
        self.factor_vol = (np.sqrt(np.diag(self.F))) # this should be F not V!

        # vol_adj_factor_exp = h.T.dot(S)*factor_vol
        try:
            self.vol_adj_factor_exposure = self.port_factor_exposure * self.factor_vol
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()

        # -4.57%, 25.5%, 10.54%

        # C) Risk Contribution (from factors) = h' * V * S * B
        self.risk_contrib_from_factors = self.h.T.dot(self.V).dot(self.S) / self.port_vol * (B)

        # D) Risk Contribution (%)
        self.risk_contrib_from_factors_pct = self.risk_contrib_from_factors / self.port_vol

        # port returns
        self.port_returns = self.h.T.dot(R)

        self.factor_returns = self.S.T.dot(R)

        # E) Return contribution
        #self.return_contrib_from_factors = self.h.T * (self.R) # this is wrong...
#        self.return_contrib_from_factors = self.R.T.dot(S)*B.dot(factor_rets )
        self.return_contrib_from_factors = B*(self.factor_returns)

        # resid portfolio
        self.u = self.h - self.S.dot(B)
        self.resid_port = self.u

        # risk contrib from resid
        self.risk_contrib_from_resid = self.h.T.dot(self.V).dot(self.u)/ self.port_vol
        # should these be split out into factors? no right? (replace * with dot)

        # return contrib from resid
        self.return_contrib_from_resid = self.u.T.dot(self.R)
        # should these be split out into factors? no right? (replace * with dot)

    def check_all_dims_align(self):
        return self.V.shape[0]==self.h.shape[0]==self.S.shape[0]==self.R.shape[0]

    def __repr__(self):
        return("""
        factorAttribution class
        =======================
        Portfolio Variance = {}
        Portfolio Vol = {}
        Portfolio Returns = {}
        
        Exposures
        =========
        Portfolio Factor Exposures = {}
        Vol Adj Factor Exposures = {}
        
        Risk Contributions
        ==================
        Risk Contrib from Factors = {}
        Risk Contrib from Factors (%) = {}
        Risk Contrib from Resid = {}
        
        Return Contributions
        ==================
        Return Contrib from Factors = {}
        Return Contrib from Resid = {}
        
        
        """.format(self.port_var,
                    self.port_vol,
                   self.port_returns,
                    self.port_factor_exposure,
                    self.vol_adj_factor_exposure,
                   self.risk_contrib_from_factors,
                   self.risk_contrib_from_factors_pct,
                   self.risk_contrib_from_resid,
                   [x for x in self.return_contrib_from_factors],
                   self.return_contrib_from_resid
                   ))



# Argo tea - Sunday , 3-24-2019
# 2 PM
##################################
import numpy as np
import pandas as pd

V = np.array([[0.0232, 0.0163, 0.0095],[0.0163, 0.2601, 0.0122],[0.0095, 0.0122, 0.0174]])

h = np.array([-0.3, 0.5, 0.8])

S = np.identity(3)

R = np.array([0.0604, 0.1795, 0.0419])

#############
# Portfolio variance = h' * V * h

port_var = h.T.dot(V).dot(h) # 7.86%
port_vol = np.sqrt(port_var) # 28.02%

# A) Factor exposure
# B = (S' * V * S)-1 * S' * V * h
B = np.linalg.inv(S.T.dot(V).dot(S)).dot( S.T.dot(V).dot(h) )
factor_exp = B
#WRONG -factor_exp = h.dot(S)

# B) Vol Adj Exposure
factor_vol = (np.sqrt(np.diag(V)))

#vol_adj_factor_exp = h.T.dot(S)*factor_vol

vol_adj_factor_exp = factor_exp * factor_vol

vol_adj_factor_exp
# -4.57%, 25.5%, 10.54%

# C) Risk Contribution (from factors) = h' * V * S * B

risk_contrib_from_factors = h.T.dot(V).dot(S)/port_vol*(B)
risk_contrib_from_factors

# D) Risk Contribution (%)
risk_contrib_from_factors_pct = risk_contrib_from_factors/port_vol
risk_contrib_from_factors_pct

# E) Return contribution

# F) risk contrib from resid?
u = h - S.T.dot(B)

h.T.dot(V)*(u)

u



return_contrib = h.T*(R)
return_contrib



factor_attrib = factorAttribution(V=V,
                 h=h,
                 S=S,
                 R=R)

factor_attrib.port_var
factor_attrib.port_vol

factor_attrib.vol_adj_factor_exposure
factor_attrib.risk_contrib_from_factors

factor_attrib.factor_exposure
factor_attrib.port_factor_exposure


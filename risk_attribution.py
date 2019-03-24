# Argo tea - Sunday , 3-24-2019
# 2 PM
##################################

V = np.array([[0.0232, 0.0163, 0.0095],[0.0163, 0.2601, 0.0122],[0.0095, 0.0122, 0.0174]])

h = np.array([-0.3, 0.5, 0.8])

S = np.identity(3)

R = np.array([0.0604, 0.1795, 0.0419])

#############

# A) Factor exposure
B =

factor_exp = h.dot(S)

# B) Vol Adj Exposure

factor_vol =


vol_adj_factor_exp = factor_exp * factor_vol

# C) Risk Contribution

port_vol = np.sqrt(h.T.dot(V).dot(h))

risk_contrib_from_factors = h.T.dot(V).dot(h).dot(B)

# D) Risk Contribution (%)

# E) Return contribution






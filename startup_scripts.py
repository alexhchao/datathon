
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report
from scipy.stats import zscore

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from ols_functions import replace_with_dummies


from ggplot import diamonds

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")  # Do this before importing pyplot!
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/49918998/plt-show-not-working-in-pycharm

#import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
pd.options.display.float_format = '{:,.4f}'.format
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

pd.options.display.max_rows = 20
pd.options.display.max_columns = 20



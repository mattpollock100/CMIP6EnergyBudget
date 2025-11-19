
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


import statsmodels.api as sm
from scipy import stats

from scipy.optimize import curve_fit
from scipy.stats import linregress
import pandas as pd

def r_squared_calc(y_true, y_pred):
    residual = y_true - y_pred
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def rmse_calc(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def aic_calc(y_true, y_pred, k):
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    return n * np.log(rss / n) + 2 * k

def pairwise_rmse(df):
    """
    Returns a DataFrame M where M.loc[x, y] is the RMSE of
    regressing column y on column x.
    """
    cols = df.columns
    # prepare the result matrix
    rmse_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

    for x in cols:
        X = df[[x]].values  # predictor must be 2D
        for y in cols:
            if x == y:
                rmse_matrix.loc[x, y] = 0.0
                continue

            # drop NaNs for the pair (optional, depending on your data)
            pair = df[[x, y]].dropna()
            X_pair = pair[[x]].values
            y_pair = pair[y].values

            # fit linear model
            model = LinearRegression().fit(X_pair, y_pair)
            y_pred = model.predict(X_pair)

            # compute RMSE
            rmse = rmse_calc(y_pair, y_pred)
            rmse_matrix.loc[x, y] = rmse

    return rmse_matrix


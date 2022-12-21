import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging
import warnings
from scipy import stats
from scipy.special import inv_boxcox
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


def make_stationary(data: pd.Series, project: str, alpha: float = 0.05):
    """
    Transforms non-stationary by log transformation and differencing

    Args:
        data: dataset
        project:
        alpha: default value = 0.05

    Returns:
        log transformed data, inverse value, stationary data & differencing order
    """

    max_diff = np.floor((len(data)/2)-1)
    boxcox_data, lmbda = stats.boxcox(data[project])
    boxcox_data = pd.Series(boxcox_data, index=data.index)

    # Test to see if the log transformed time series is stationary
    if (sm.tsa.stattools.adfuller(boxcox_data)[1] < alpha) & (sm.tsa.stattools.kpss(boxcox_data)[1] > alpha):
        return boxcox_data, lmbda, None, None

    p_values = []

    # Test for differencing orders from 1 to max_diff order
    for i in range(1, max_diff):
        # Perform ADF test
        result = sm.tsa.stattools.adfuller(boxcox_data.diff(i).dropna())
        # Append p-value
        p_values.append((i,result[1]))

    # Keep only those where p-value is lower than significance level
    significant = [p for p in p_values if p[1] < alpha]
    # Sort by the differencing order
    if significant:
        significant = sorted(significant, key=lambda x: x[0])

        # get the differencing order
        diff_order = significant[0][0]

        # make the time series stationary
        stationary_data = boxcox_data.diff(diff_order).dropna()

        return boxcox_data, lmbda, stationary_data, diff_order

    else:
        return boxcox_data, lmbda, None, None

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


def lagged_correlation(boxcox_data: pd.DataFrame, differenced_data: pd.DataFrame) -> tuple:
    """
    Function to calculate highly correlated lags - pacf & acf

    Args:
        boxcox_data: log transformed data
        differenced_data: stationary data

    Returns:
        Highly correlated lags
    """

    if differenced_data is not None:
        transformed_data = differenced_data.copy()
    else:
        transformed_data = boxcox_data.copy()

    corr_pacf = sm.tsa.stattools.pacf(transformed_data)
    range_pacf = np.arange(len(corr_pacf))
    corr_pacf = list(zip(range_pacf, corr_pacf))

    pacf_data = []
    for item in corr_pacf:
        if (abs(item[1]) > .25) & (item[0] != 0):
            pacf_data.append(item)

    corr_acf = sm.tsa.stattools.acf(transformed_data)
    range_acf = np.arange(len(corr_acf))
    corr_acf = list(zip(range_acf, corr_acf))

    acf_data = []
    for item in corr_acf:
        if (abs(item[1]) > .25) & (item[0] != -0):
            acf_data.append(item)

    return pacf_data, acf_data


def auto_regressive(
        pacf: list,
        boxcox_data: pd.DataFrame,
        stationary_data: pd.DataFrame,
        test_len: int,
        data: pd.DataFrame,
        diff_order: int,
        lmbda: float,
        project: str
):
    """
    Function to train autoregressive model

    Args:
        pacf: partially auto correlated values
        boxcox_data: log transformed data
        stationary_data: difference data
        test_len: length of data for testing
        data: dataset
        diff_order: level of differencing performed
        lmbda: inverse value
        project: name of prediction column

    Return:
        autoregressive model object along with its performance metrics
    """

    try:
        performance = []
        for items in pacf:
            if diff_order:
                ar = sm.tsa.ARIMA(stationary_data, order=(items[0], 0, 0)).fit()
                prediction = data.copy()
                prediction['ar_boxcox_diff'] = ar.predict(stationary_data.index.min(), stationary_data.index.max())
                prediction['ar_boxcox'] = prediction['ar_boxcox_diff'].cumsum()
                count = 0
                while count < diff_order:
                    prediction['ar_boxcox'] = prediction['ar_boxcox'].add(boxcox_data[count])
                    count += 1

            else:
                ar = sm.tsa.ARIMA(boxcox_data, order=(items[0], 0, 0)).fit()
                prediction = data.copy()
                prediction['ar_boxcox'] = ar.predict(boxcox_data.index.min(), boxcox_data.index.max())

            prediction['ar'] = inv_boxcox(prediction['ar_boxcox'], lmbda)
            rmse = np.sqrt(mean_squared_error(data[project][-test_len:], prediction['ar'][-test_len:])).round(2)
            mape = np.round(
                np.mean(
                    np.abs(data[project][-test_len:] - prediction['ar'][-test_len:])/
                    data[project][-test_len:]
                ) * 100, 2
            )

            performance.append((items[0], rmse, mape))
        performance = sorted(performance, key=lambda x: x[1])
        if (diff_order and performance):
            arm = sm.tsa.ARIMA(stationary_data, order=(performance[0][0], 0, 0)).fit()
            return pd.DataFrame(
                {
                    'Object': arm,
                    'Method': ['auto_regressive'],
                    'rmse': [rmse],
                    'mape': [mape],
                    'p': [performance[0][0]],
                    'd': [0],
                    'q': [0],
                    'P': [0],
                    'D': [0],
                    'Q': [0]
                }
            )

        elif performance:
            arm = sm.tsa.ARIMA(boxcox_data, order=(performance[0][0], 0, 0)).fit()
            return pd.DataFrame(
                {
                    'Object': arm,
                    'Method': ['auto_regressive'],
                    'rmse': [rmse],
                    'mape': [mape],
                    'p': [performance[0][0]],
                    'd': [0],
                    'q': [0],
                    'P': [0],
                    'D': [0],
                    'Q': [0]
                }
            )

        else:
            return pd.DataFrame(
                {
                    'Object': None,
                    'Method': ['auto_regressive'],
                    'rmse': [np.inf],
                    'mape': [np.inf],
                    'p': [0],
                    'd': [0],
                    'q': [0],
                    'P': [0],
                    'D': [0],
                    'Q': [0]
                }
            )

    except BaseException:
        logging.exception('Error encountered in auto_regressive!')
        return pd.DataFrame(
            {
                'Object': None,
                'Method': ['auto_regressive'],
                'rmse': [np.inf],
                'mape': [np.inf],
                'p': [0],
                'd': [0],
                'q': [0],
                'P': [0],
                'D': [0],
                'Q': [0]
            }
        )



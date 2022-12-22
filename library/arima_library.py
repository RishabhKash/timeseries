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
    Function to train autoregressive algorithm

    Args:
        pacf: partially auto correlated lags
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

    logging.info("AR modelling!")
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
        if diff_order and performance:
            arm = sm.tsa.ARIMA(stationary_data, order=(performance[0][0], 0, 0)).fit()
            return pd.DataFrame(
                {
                    'Object': arm,
                    'Method': ['auto_regressive'],
                    'rmse': [rmse],
                    'mape': [mape],
                    'p': [performance[0][0]],
                    'd': [diff_order],
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

    except Exception as e:
        logging.error("Error while training Auto_Regressive: %s", e)
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


def moving_average(
        acf: list,
        boxcox_data: pd.DataFrame,
        stationary_data: pd.DataFrame,
        test_len: int,
        data: pd.DataFrame,
        diff_order: int,
        lmbda: float,
        project: str
):
    """
    Function to train moving average algorithm

    Args:
        acf: auto correlated lags
        boxcox_data: log transformed data
        stationary_data: difference data
        test_len: length of data for testing
        data: dataset
        diff_order: level of differencing performed
        lmbda: inverse value
        project: name of prediction column

    Return:
        moving average model object along with its performance metrics
    """

    logging.info("MA modelling!")
    try:
        performance = []
        for items in acf:
            if diff_order:
                ma = sm.tsa.ARIMA(stationary_data, order=(0, 0, items[0])).fit()
                prediction = data.copy()
                prediction['ma_boxcox_diff'] = ma.predict(data.index[diff_order], data.index.max())
                prediction['ma_boxcox'] = prediction['ma_boxcox_diff'].cumsum()
                count = 0
                while count < diff_order:
                    prediction['ma_boxcox'] = prediction['ma_boxcox'].add(boxcox_data[count])
                    count += 1

            else:
                ma = sm.tsa.ARIMA(boxcox_data, order=(0, 0, items[0])).fit()
                prediction = data.copy()
                prediction['ma_boxcox'] = ma.predict(data.index.min(), data.index.max())

            prediction['ma'] = inv_boxcox(prediction['ma_boxcox'], lmbda)
            rmse = np.sqrt(mean_squared_error(data[project][-test_len:], prediction['ma'][-test_len:])).round(2)
            mape = np.round(
                np.mean(
                    np.abs(data[project][-test_len:] - prediction['ma'][-test_len:])/
                    data[project][-test_len:]
                ) * 100, 2
            )

            performance.append((items[0], rmse, mape))
        performance = sorted(performance, key=lambda x: x[1])
        if diff_order and performance:
            movingaverage = sm.tsa.ARIMA(stationary_data, order=(0, 0, performance[0][0])).fit()
            return pd.DataFrame(
                {
                    'Object': movingaverage,
                    'Method': ['moving_average'],
                    'rmse': [rmse],
                    'mape': [mape],
                    'p': [0],
                    'd': [diff_order],
                    'q': [performance[0][0]],
                    'P': [0],
                    'D': [0],
                    'Q': [0]
                }
            )

        elif performance:
            movingaverage = sm.tsa.ARIMA(boxcox_data, order=(0, 0, performance[0][0])).fit()
            return pd.DataFrame(
                {
                    'Object': movingaverage,
                    'Method': ['moving_average'],
                    'rmse': [rmse],
                    'mape': [mape],
                    'p': [0],
                    'd': [0],
                    'q': [performance[0][0]],
                    'P': [0],
                    'D': [0],
                    'Q': [0]
                }
            )

        else:
            return pd.DataFrame(
                {
                    'Object': None,
                    'Method': ['moving_average'],
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

    except Exception as e:
        logging.error("Error while training Moving_Average: %s", e)
        return pd.DataFrame(
            {
                'Object': None,
                'Method': ['moving_average'],
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


def arma(
        pacf: list,
        acf: list,
        boxcox_data: pd.DataFrame,
        stationary_data: pd.DataFrame,
        test_len: int,
        data: pd.DataFrame,
        diff_order: int,
        lmbda: float,
        project: str
):
    """
    Function to train Autoregressive Moving Average algorithm

    Args:
        pacf: partially auto correlated lags
        acf: auto correlated lags
        boxcox_data: log transformed data
        stationary_data: difference data
        test_len: length of data for testing
        data: dataset
        diff_order: level of differencing performed
        lmbda: inverse value
        project: name of prediction column

    Return:
        arma model object along with its performance metrics
    """

    try:
        performance = []
        for items in pacf:
            for levels in acf:
                if diff_order:
                    arma = sm.tsa.ARIMA(stationary_data, order=(items[0], 0, levels[0])).fit()
                    prediction = data.copy()
                    prediction['arma_boxcox_diff'] = arma.predict(data.index[diff_order], data.index.max())
                    prediction['arma_boxcox'] = prediction['arma_boxcox_diff'].cumsum()
                    count = 0
                    while count < diff_order:
                        prediction['arma_boxcox'] = prediction['arma_boxcox'].add(boxcox_data[count])
                        count += 1

                else:
                    arma = sm.tsa.ARIMA(boxcox_data, order=(items[0], 0, levels[0])).fit()
                    prediction = data.copy()
                    prediction['arma_boxcox'] = arma.predict(data.index.min(), data.index.max())

                prediction['arma'] = inv_boxcox(prediction['arma_boxcox'], lmbda)
                rmse = np.sqrt(mean_squared_error(data[project][-test_len:], prediction['arma'][-test_len:])).round(2)
                mape = np.round(
                    np.mean(
                        np.abs(data[project][-test_len:] - prediction['arma'][-test_len:]) /
                        data[project][-test_len:]
                    ) * 100, 2
                )

            performance.append((items[0], levels[0], rmse, mape))
        performance = sorted(performance, key=lambda x: x[2])
        if diff_order and performance:
            ar_ma = sm.tsa.ARIMA(stationary_data, order=(performance[0][0], 0, performance[0][1])).fit()
            return pd.DataFrame(
                {
                    'Object': ar_ma,
                    'Method': ['ar_ma'],
                    'rmse': [rmse],
                    'mape': [mape],
                    'p': [performance[0][0]],
                    'd': [diff_order],
                    'q': [performance[0][1]],
                    'P': [0],
                    'D': [0],
                    'Q': [0]
                }
            )

        elif performance:
            ar_ma = sm.tsa.ARIMA(boxcox_data, order=(performance[0][0], 0, performance[0][1])).fit()
            return pd.DataFrame(
                {
                    'Object': ar_ma,
                    'Method': ['ar_ma'],
                    'rmse': [rmse],
                    'mape': [mape],
                    'p': [performance[0][0]],
                    'd': [0],
                    'q': [performance[0][1]],
                    'P': [0],
                    'D': [0],
                    'Q': [0]
                }
            )

        else:
            return pd.DataFrame(
                {
                    'Object': None,
                    'Method': ['ar_ma'],
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

    except Exception as e:
        logging.error("Error while training ARMA: %s", e)
        return pd.DataFrame(
            {
                'Object': None,
                'Method': ['ar_ma'],
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


def arima(
        boxcox_data: pd.DataFrame,
        train_len: int,
        data: pd.DataFrame,
        lmbda: float,
        project: str
):
    """
    Function to train arima/sarima algorithm

    Args:
        boxcox_data: log transformed data
        train_len: length of data for training
        data: dataset
        lmbda: inverse value
        project: name of prediction column

    Returns:
        arima/sarima model object along with its performance metrics
    """

    try:
        max_diff = np.floor((len(boxcox_data)/2)-1)
        stepwise_model = auto_arima(
            boxcox_data[:train_len], start_p=0, start_q=0, max_p=30, max_d=max_diff, max_q=30, m=12,
            start_P=0, start_Q=0, max_P=30, max_Q=30, max_D=max_diff, seasonal=True, trace=True,
            error_action='ignore', suppress_warnings=True, stepwise=True, n_jobs=-1
        )

        prediction = stepwise_model.predict(len(boxcox_data) - train_len)
        prediction = inv_boxcox(prediction, lmbda)
        rmse = np.sqrt(mean_squared_error(data[project][train_len:], prediction)).round(2)
        mape = np.round(
            np.mean(
                np.abs(data[project][train_len:] - prediction)/
                data[project][train_len:]
            ) * 100, 2
        )

        stepwise_model = auto_arima(
            boxcox_data, start_p=0, start_q=0, max_p=30, max_d=max_diff, max_q=30, m=12,
            start_P=0, start_Q=0, max_P=30, max_Q=30, max_D=max_diff, seasonal=True, trace=True,
            error_action='ignore', suppress_warnings=True, stepwise=True, n_jobs=-1
        )
        return pd.DataFrame(
            {
                'Object': stepwise_model,
                'Method': ['ARIMA'],
                'RMSE': [rmse],
                'MAPE': [mape],
                'p': [(stepwise_model.get_params()).get('order')[0]],
                'd': [(stepwise_model.get_params()).get('order')[1]],
                'q': [(stepwise_model.get_params()).get('order')[2]],
                'P': [(stepwise_model.get_params()).get('seasonal_order')[0]],
                'D': [(stepwise_model.get_params()).get('seasonal_order')[1]],
                'Q': [(stepwise_model.get_params()).get('seasonal_order')[2]]
            }
        )
    except Exception as e:
        logging.error("Error while training ARIMA/SARIMA: %s", e)
        return pd.DataFrame(
            {
                'Object': None,
                'Method': ['ARIMA'],
                'RMSE': [np.inf],
                'MAPE': [np.inf],
                'p': [0],
                'd': [0],
                'q': [0],
                'P': [0],
                'D': [0],
                'Q': [0]
            }
        )

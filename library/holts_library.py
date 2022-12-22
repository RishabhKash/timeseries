import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")


def metric_calculator(object_name: object, test_len: int, data: pd.DataFrame, project: str) -> tuple:
    """ Forecasts basis given object to calculate metrics - rmse and mape

    Args:
        object_name: model object
        test_len: length of data for validation of predictions
        data: dataset
        project: name of prediction column

    Returns:
        RMSE & MAPE score of the model, infinity in case of error
    """

    try:
        prediction = object_name.forecast(test_len)
        prediction = pd.DataFrame(prediction)
        prediction = prediction.rename(columns={0: 'Predicted'})
        rmse = np.sqrt(mean_squared_error(data[project][-test_len:], prediction['Predicted'])).round(2)
        mape = np.round(
            np.mean(
                np.abs(data[project][-test_len:] - prediction['Predicted']) / data[project][-test_len:]
            ) * 100, 2
        )

        return rmse, mape

    except Exception as e:
        logging.error('Error occurred while metric calculation: %s', e)
        return np.inf, np.inf


def simple_exponential(data: pd.DataFrame, train_len: int, test_len: int, project: str) -> pd.DataFrame:
    """
    Returns a simple exponential model along with its performance metrics

    Args:
        data: dataset
        train_len: length of data for training
        test_len: length of data for testing
        project: name of prediction column

    Returns:
        dataframe containing model object & none in case of error,
        method name, metrics - rmse & mape & infinity in case of error,
        lastly p,d,q,P,D,Q values as 0
    """

    try:
        simple_exponential_smoothing = sm.tsa.SimpleExpSmoothing(
            data[project][:train_len]
        ).fit(optimized=True)

        rmse, mape = metric_calculator(
            object_name=simple_exponential_smoothing,
            test_len=test_len,
            data=data,
            project=project
        )

        simple_exponential_smoothing = sm.tsa.SimpleExpSmoothing(
            data[project]
        ).fit(optimized=True)

        logging.info("simple_exponential complete!")

        return pd.DataFrame(
            {
                'Object': [simple_exponential_smoothing],
                'Method': ['simple_exponential_smoothing'],
                'RMSE': [rmse],
                'MAPE': [mape],
                'p': [0],
                'd': [0],
                'q': [0],
                'P': [0],
                'D': [0],
                'Q': [0]
            }
        )

    except Exception as e:
        logging.error("Error while training simple_exponential_smoothing: %s", e)
        return pd.DataFrame(
            {
                'Object': [None],
                'Method': ['simple_exponential_smoothing'],
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


def additive_trend(data: pd.DataFrame, train_len: int, test_len: int, project: str) -> pd.DataFrame:
    """
    Returns an additive trend exponential model along with its performance metrics

    Args:
        data: dataset
        train_len: length of data for training
        test_len: length of data for testing
        project: name of prediction column

    Returns:
        dataframe containing model object & none in case of error,
        method name, metrics - rmse & mape & infinity in case of error,
        lastly p,d,q,P,D,Q values as 0
    """

    try:
        exponential_smoothing_add = sm.tsa.ExponentialSmoothing(
            data[project][:train_len], seasonal_periods=12, trend='add', seasonal=None
        ).fit(optimized=True)

        rmse, mape = metric_calculator(
            object_name=exponential_smoothing_add,
            test_len=test_len,
            data=data,
            project=project
        )

        exponential_smoothing_add = sm.tsa.ExponentialSmoothing(
            data[project], seasonal_periods=12, trend='add', seasonal=None
        ).fit(optimized=True)

        logging.info("additive_trend complete!")

        return pd.DataFrame(
            {
                'Object': [exponential_smoothing_add],
                'Method': ['exponential_smoothing_add'],
                'RMSE': [rmse],
                'MAPE': [mape],
                'p': [0],
                'd': [0],
                'q': [0],
                'P': [0],
                'D': [0],
                'Q': [0]
            }
        )

    except Exception as e:
        logging.error("Error while training additive_trend_holts: %s", e)
        return pd.DataFrame(
            {
                'Object': [None],
                'Method': ['exponential_smoothing_add'],
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


def multiplicative_trend(data: pd.DataFrame, train_len: int, test_len: int, project: str) -> pd.DataFrame:
    """
    Returns a multiplicative trend exponential model along with its performance metrics

    Args:
        data: dataset
        train_len: length of data for training
        test_len: length of data for testing
        project: name of prediction column

    Returns:
        dataframe containing model object & none in case of error,
        method name, metrics - rmse & mape & infinity in case of error,
        lastly p,d,q,P,D,Q values as 0
    """

    try:
        exponential_smoothing_mul = sm.tsa.ExponentialSmoothing(
            data[project][:train_len], seasonal_periods=12, trend='mul', seasonal=None
        ).fit(optimized=True)

        rmse, mape = metric_calculator(
            object_name=exponential_smoothing_mul,
            test_len=test_len,
            data=data,
            project=project
        )

        exponential_smoothing_mul = sm.tsa.ExponentialSmoothing(
            data[project], seasonal_periods=12, trend='mul', seasonal=None
        ).fit(optimized=True)

        logging.info("multiplicative_trend complete!")

        return pd.DataFrame(
            {
                'Object': [exponential_smoothing_mul],
                'Method': ['exponential_smoothing_mul'],
                'RMSE': [rmse],
                'MAPE': [mape],
                'p': [0],
                'd': [0],
                'q': [0],
                'P': [0],
                'D': [0],
                'Q': [0]
            }
        )

    except Exception as e:
        logging.error("Error while training multiplicative_trend_holts: %s", e)
        return pd.DataFrame(
            {
                'Object': [None],
                'Method': ['exponential_smoothing_mul'],
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


def a_a_seasonality(data: pd.DataFrame, train_len: int, test_len: int, project: str) -> pd.DataFrame:
    """
    Returns additive trend & seasonality exponential model along with its performance metrics

    Args:
        data: dataset
        train_len: length of data for training
        test_len: length of data for testing
        project: name of prediction column

    Returns:
        dataframe containing model object & none in case of error,
        method name, metrics - rmse & mape & infinity in case of error,
        lastly p,d,q,P,D,Q values as 0
    """

    try:
        exponential_smoothing_seasonality_a_a = sm.tsa.ExponentialSmoothing(
            data[project][:train_len], seasonal_periods=12, trend='add', seasonal='add'
        ).fit(optimized=True)

        rmse, mape = metric_calculator(
            object_name=exponential_smoothing_seasonality_a_a,
            test_len=test_len,
            data=data,
            project=project
        )

        exponential_smoothing_seasonality_a_a = sm.tsa.ExponentialSmoothing(
            data[project], seasonal_periods=12, trend='add', seasonal='add'
        ).fit(optimized=True)

        logging.info("a_a_seasonality complete!")

        return pd.DataFrame(
            {
                'Object': [exponential_smoothing_seasonality_a_a],
                'Method': ['exponential_smoothing_seasonality_a_a'],
                'RMSE': [rmse],
                'MAPE': [mape],
                'p': [0],
                'd': [0],
                'q': [0],
                'P': [0],
                'D': [0],
                'Q': [0]
            }
        )

    except Exception as e:
        logging.error("Error while training a_a_seasonality: %s", e)
        return pd.DataFrame(
            {
                'Object': [None],
                'Method': ['exponential_smoothing_seasonality_a_a'],
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


def a_m_seasonality(data: pd.DataFrame, train_len: int, test_len: int, project: str) -> pd.DataFrame:
    """
    Returns additive trend & multiplicative seasonality exponential model along with its performance metrics

    Args:
        data: dataset
        train_len: length of data for training
        test_len: length of data for testing
        project: name of prediction column

    Returns:
        dataframe containing model object & none in case of error,
        method name, metrics - rmse & mape & infinity in case of error,
        lastly p,d,q,P,D,Q values as 0
    """

    try:
        exponential_smoothing_seasonality_a_m = sm.tsa.ExponentialSmoothing(
            data[project][:train_len], seasonal_periods=12, trend='add', seasonal='mul'
        ).fit(optimized=True)

        rmse, mape = metric_calculator(
            object_name=exponential_smoothing_seasonality_a_m,
            test_len=test_len,
            data=data,
            project=project
        )

        exponential_smoothing_seasonality_a_m = sm.tsa.ExponentialSmoothing(
            data[project], seasonal_periods=12, trend='add', seasonal='mul'
        ).fit(optimized=True)

        logging.info("a_m_seasonality complete!")

        return pd.DataFrame(
            {
                'Object': [exponential_smoothing_seasonality_a_m],
                'Method': ['exponential_smoothing_seasonality_a_m'],
                'RMSE': [rmse],
                'MAPE': [mape],
                'p': [0],
                'd': [0],
                'q': [0],
                'P': [0],
                'D': [0],
                'Q': [0]
            }
        )

    except Exception as e:
        logging.error("Error while training a_m_seasonality: %s", e)
        return pd.DataFrame(
            {
                'Object': [None],
                'Method': ['exponential_smoothing_seasonality_a_m'],
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


def m_a_seasonality(data: pd.DataFrame, train_len: int, test_len: int, project: str) -> pd.DataFrame:
    """
    Returns multiplicative trend & additive seasonality exponential model along with its performance metrics

    Args:
        data: dataset
        train_len: length of data for training
        test_len: length of data for testing
        project: name of prediction column

    Returns:
        dataframe containing model object & none in case of error,
        method name, metrics - rmse & mape & infinity in case of error,
        lastly p,d,q,P,D,Q values as 0
    """

    try:
        exponential_smoothing_seasonality_m_a = sm.tsa.ExponentialSmoothing(
            data[project][:train_len], seasonal_periods=12, trend='mul', seasonal='add'
        ).fit(optimized=True)

        rmse, mape = metric_calculator(
            object_name=exponential_smoothing_seasonality_m_a,
            test_len=test_len,
            data=data,
            project=project
        )

        exponential_smoothing_seasonality_m_a = sm.tsa.ExponentialSmoothing(
            data[project], seasonal_periods=12, trend='mul', seasonal='add'
        ).fit(optimized=True)

        logging.info("m_a_seasonality complete!")

        return pd.DataFrame(
            {
                'Object': [exponential_smoothing_seasonality_m_a],
                'Method': ['exponential_smoothing_seasonality_m_a'],
                'RMSE': [rmse],
                'MAPE': [mape],
                'p': [0],
                'd': [0],
                'q': [0],
                'P': [0],
                'D': [0],
                'Q': [0]
            }
        )

    except Exception as e:
        logging.error("Error while training m_a_seasonality: %s", e)
        return pd.DataFrame(
            {
                'Object': [None],
                'Method': ['exponential_smoothing_seasonality_m_a'],
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


def m_m_seasonality(data: pd.DataFrame, train_len: int, test_len: int, project: str) -> pd.DataFrame:
    """
    Returns multiplicative trend & seasonality exponential model along with its performance metrics

    Args:
        data: dataset
        train_len: length of data for training
        test_len: length of data for testing
        project: name of prediction column

    Returns:
        dataframe containing model object & none in case of error,
        method name, metrics - rmse & mape & infinity in case of error,
        lastly p,d,q,P,D,Q values as 0
    """

    try:
        exponential_smoothing_seasonality_m_m = sm.tsa.ExponentialSmoothing(
            data[project][:train_len], seasonal_periods=12, trend='mul', seasonal='mul'
        ).fit(optimized=True)

        rmse, mape = metric_calculator(
            object_name=exponential_smoothing_seasonality_m_m,
            test_len=test_len,
            data=data,
            project=project
        )

        exponential_smoothing_seasonality_m_m = sm.tsa.ExponentialSmoothing(
            data[project], seasonal_periods=12, trend='mul', seasonal='mul'
        ).fit(optimized=True)

        logging.info("m_m_seasonality complete!")

        return pd.DataFrame(
            {
                'Object': [exponential_smoothing_seasonality_m_m],
                'Method': ['exponential_smoothing_seasonality_m_m'],
                'RMSE': [rmse],
                'MAPE': [mape],
                'p': [0],
                'd': [0],
                'q': [0],
                'P': [0],
                'D': [0],
                'Q': [0]
            }
        )

    except Exception as e:
        logging.error("Error while training m_m_seasonality: %s", e)
        return pd.DataFrame(
            {
                'Object': [None],
                'Method': ['exponential_smoothing_seasonality_m_m'],
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

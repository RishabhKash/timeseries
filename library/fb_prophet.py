import logging
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error


def ideal_prophet(data: pd.DataFrame, train_len: int, test_len: int, project: str):
    """
    Function to train Facebook's prophet algorithm
    """

    try:
        # data = data.reset_index()
        data = data.rename(columns={project: "y", data.columns[0]: "ds"})
        m = Prophet(daily_seasonality=False, weekly_seasonality=False)
        model = m.fit(data[:train_len])
        future = model.make_future_dataframe(periods=test_len, freq="M")
        forecast = model.predict(future)

        rmse = np.sqrt(mean_squared_error(data['y'][-test_len:], forecast['yhat'][-test_len:])).round(2)
        mape = np.round(
            np.mean(
                np.abs(data['y'][-test_len:].values - forecast['yhat'][-test_len:].values) /
                data['y'][-test_len:].values
            ) * 100, 2
        )

        m1 = Prophet(daily_seasonality=False, weekly_seasonality=False)
        fb_prophet = m1.fit(data)

        return pd.DataFrame(
            {
                'Object': [fb_prophet],
                'Method': ['fbprophet'],
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
        logging.error("Error while training fb_prophet: %s", e)
        return pd.DataFrame(
            {
                'Object': [None],
                'Method': ['fbprophet'],
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

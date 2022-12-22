import logging
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime
from scipy.special import inv_boxcox
from library.fb_prophet import ideal_prophet
from library.holts_library import additive_trend, multiplicative_trend, a_a_seasonality, a_m_seasonality, \
    m_a_seasonality, m_m_seasonality
from library.arima_library import make_stationary, lagged_correlation, auto_regressive, arma, arima


def train_predict(
        processed_data: pd.DataFrame,
        project: str
):
    """
    Tests an array of time series algorithm and predicts for the next six months using the best performing model

    Args:
        processed_data: dataset
        project: name of prediction column

    Returns:
        prediction for the next six months
    """

    data = processed_data.copy()
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
    data.iloc[:, 0] = data.iloc[:, 0].apply(lambda x: x + relativedelta(day=31))
    start_date = datetime.now().strftime("%Y-%m")
    end_date = (datetime.now() + relativedelta(months=5)).strftime("%Y-%m")
    model_performance = pd.DataFrame()
    data.loc[data[project] <= 0].interpolate(limit_direction="both")

    # Deciding test length basis total available data
    if (len(data) >= 3) and (len(data) <= 5):
        test_len = 1
    elif len(data) > 5:
        test_len = 3
    else:
        return "Insufficient data for prediction"

    train_len = len(data) - test_len
    performance = pd.DataFrame()
    logging.info(f" train length {train_len} test length {test_len}")

    # training different time series models
    outcome = ideal_prophet(data=data, train_len=train_len, test_len=test_len, project=project)
    performance = pd.concat([performance, outcome])

    outcome = additive_trend(data=data, train_len=train_len, test_len=test_len, project=project)
    performance = pd.concat([performance, outcome])

    outcome = multiplicative_trend(data=data, train_len=train_len, test_len=test_len, project=project)
    performance = pd.concat([performance, outcome])

    outcome = a_a_seasonality(data=data, train_len=train_len, test_len=test_len, project=project)
    performance = pd.concat([performance, outcome])

    outcome = a_m_seasonality(data=data, train_len=train_len, test_len=test_len, project=project)
    performance = pd.concat([performance, outcome])

    outcome = m_a_seasonality(data=data, train_len=train_len, test_len=test_len, project=project)
    performance = pd.concat([performance, outcome])

    outcome = m_m_seasonality(data=data, train_len=train_len, test_len=test_len, project=project)
    performance = pd.concat([performance, outcome])

    # data preparation for ARIMA family of algorithms
    boxcox_data, lmbda, stationary_data, diff_order = make_stationary(data=data, project=project)
    pacf, acf = lagged_correlation(boxcox_data=boxcox_data, differenced_data=stationary_data)

    outcome = auto_regressive(
        pacf=pacf,
        boxcox_data=boxcox_data,
        stationary_data=stationary_data,
        test_len=test_len,
        data=data,
        diff_order=diff_order,
        lmbda=lmbda,
        project=project
    )
    performance = pd.concat([performance, outcome])

    outcome = arma(
        pacf=pacf,
        acf=acf,
        boxcox_data=boxcox_data,
        stationary_data=stationary_data,
        test_len=test_len,
        data=data,
        diff_order=diff_order,
        lmbda=lmbda,
        project=project
    )
    performance = pd.concat([performance, outcome])

    outcome = arima(
        boxcox_data=boxcox_data,
        train_len=train_len,
        data=data,
        lmbda=lmbda,
        project=project
    )
    if ((outcome['p']) | (outcome['P'])).any():
        performance = pd.concat([performance, outcome])

    performance = performance.sort_values(by=['RMSE', 'MAPE'])
    model_performance = pd.concat([model_performance, performance])

    # Prediction for next six months
    if performance.iloc[0][1] in ('simple_exponential_smoothing', 'exponential_smoothing_add',
                                  'exponential_smoothing_mul', 'exponential_smoothing_seasonality_a_a',
                                  'exponential_smoothing_seasonality_a_m', 'exponential_smoothing_seasonality_m_a',
                                  'exponential_smoothing_seasonality_m_m'):
        result = performance.iloc[0][0].predict(start_date, end_date)
        result = pd.DataFrame(result)
        result.columns = ['Predicted']

    elif performance.iloc[0][1] in ('auto_regressive', 'moving_average', 'ar_ma'):
        if performance.iloc[0][5] > 0:
            result = performance.iloc[0][0].predict(stationary_data.index.min(), end_date)
            result = pd.DataFrame(result)
            result.columns = ['Predicted']
            result['Predicted'] = result['Predicted'].cumsum()
            count = 0
            while count < performance.iloc[0][5]:
                result['Predicted'] = result['Predicted'].add(boxcox_data[count])
                count += 1

        else:
            print(boxcox_data.index.min(), end_date)
            print(boxcox_data.head())
            result = performance.iloc[0][0].predict(boxcox_data.index.min(), end_date)
            result = pd.DataFrame(result)
            result.columns = ['Predicted']

        result['Predicted'] = inv_boxcox(result['Predicted'], lmbda)
        result = result[-6:]

    elif performance[0][1] in ('fbprophet'):
        temp = performance.iloc[0][0].make_future_dataframe(periods=6, freq="M")
        future = performance.iloc[0][0].predict(temp)
        result = future[["ds", "yhat"]][-6:]
        result.set_index("ds", inplace=True)

    else:
        result = performance.iloc[0][0].predict(6)
        result = pd.DataFrame(result)
        result.columns = ['Predicted']
        result['Predicted'] = inv_boxcox(result['Predicted'], lmbda)

    result = round(result, 0).abs()

    return model_performance, result

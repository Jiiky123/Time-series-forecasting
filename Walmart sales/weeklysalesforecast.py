import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
from statistics import mean
import os
os.chdir('D:/PythonProjektATOM/Git/Repositories/Time-series-forecasting/Walmart sales/')

# importing prediction_df whick includes aggregated sales
prediction_df = pd.read_csv('prediction_df.csv', sep=',', header=None, index_col=0)

prediction_df.index.name = 'ds'
prediction_df.columns = ['y']
prediction_df.index = pd.to_datetime(prediction_df.index, format="%Y/%m/%d")


# SARIMAX AGG. SALES FORECAST

# test & train AGG SALES
train_length = 80  # weekly data
train_df = prediction_df.iloc[:train_length]
test_df = prediction_df.iloc[train_length:len(prediction_df)]

train_series = train_df
test_series = test_df


def plot_acf_pacf(lags):  # ACF and PACF residual plots

    plot_acf(prediction_df, lags=lags)
    plot_pacf(prediction_df, lags=lags)
    plt.show()


def SARIMAX_fit():  # fit sarimax model (p,d,q)(P,D,Q,m)

    model = SARIMAX(train_series, order=(0, 1, 0), seasonal_order=(1, 0, 0, 52))
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(train_series), len(prediction_df)-1, typ='levels')
    error = mean_squared_error(test_series, yhat)
    print('Test MSE: %.3f' % error)


def plot_result():  # out of sample result

    plt.plot(test_series)
    plt.plot(yhat, color='red')
    plt.show()


# FB PROPHET AGG. SALES FORECAST
def Prophet_fit():
    # include arbitrary index
    train_series = train_series.reset_index()

    # fit prophet model
    yearly_seasonality = 30
    m = Prophet(yearly_seasonality=yearly_seasonality)
    m.fit(train_series)

    # make prediction dataframe
    future = m.make_future_dataframe(freq='W', periods=len(prediction_df)-train_length)
    forecast = m.predict(future)

    # plot results
    fig1 = m.plot(forecast)
    plt.plot(test_series, c='black', alpha=0.5)
    # vertical line at cutoff
    plt.axvline(x=train_series.iloc[-1, 0], c='r')
    plt.show()


# SALES BY STORE AND DEPARTMENT


# importing merged_feat_train dataset which includes store/dept sales
merged_feat_train = pd.read_csv('merged_feat_train.csv', sep=',', index_col=0)


# make a df with store as column and weekly sales as rows
stores_df = merged_feat_train[['Store', 'Dept', 'Weekly_Sales']]
stores_df = stores_df.pivot_table(values='Weekly_Sales',
                                  index=stores_df.index, columns='Store')
stores_df.index = pd.to_datetime(stores_df.index, format="%Y/%m/%d")

stores_train = stores_df.iloc[:train_length]
stores_train.index = pd.DatetimeIndex(stores_train.index.values,
                                      freq=stores_train.index.inferred_freq)
stores_test = stores_df.iloc[train_length:len(stores_df)]

# for prophet
stores_train_prophet = stores_train
stores_train_prophet.index.name = 'ds'

# naming columns (needed for fb prophet) - find a better solution?
stores_train_prophet.columns = ['y' for i in stores_train_prophet.columns]

# create arbitrary index for prophet
stores_train_prophet = stores_train_prophet.reset_index()


def predict_all_stores():

    predicted_stores = pd.DataFrame(columns=range(1, len(stores_df.columns)+1))
    for store in range(1, len(stores_df.columns)+1):
        model = SARIMAX(stores_train.iloc[:, store-1], order=(0, 1, 0),
                        seasonal_order=(1, 0, 0, 52))
        model_fit = model.fit(disp=False)
        yhat = pd.DataFrame(model_fit.predict(len(stores_train),
                                              len(stores_df)-1, typ='levels'))
        predicted_stores[store] = yhat[0]
        print(store, '/', len(stores_df.columns+1))

    predicted_stores.to_csv('predicted_stores.csv')
    print('saved')


# PLOT SARIMA PRED VS ACTUAL
predictions = pd.read_csv('predicted_stores.csv', sep=',', index_col=0)
# predictions = predictions.rename(columns={'0': '45'})
predictions.index = pd.to_datetime(predictions.index, format="%Y/%m/%d")

# optimistic / pessimistic predictions
predictions_best_perf = pd.Series([(predictions.iloc[-1, store]-predictions.iloc[0, store]) /
                                   predictions.iloc[0, store]
                                   for store in range(len(predictions.columns))])
predictions_best_perf.index = np.arange(1, len(predictions_best_perf)+1)


def plot_predictions_vs_actual_sarima():  # plotterinho

    for i in range(len(predictions.columns)):
        plt.plot(predictions.iloc[:, i], c='r')
        plt.plot(stores_test.iloc[:, i], c='b')
        plt.show()


def mse_sarima():  # mean square errors

    MSE = []
    for store in range(len(predictions.columns)):
        error = mean_squared_error(
            stores_test.iloc[:, store], predictions.iloc[:, store])
        MSE.append(error)
        # print(store+1, ' MSE: ', error)
    MSE = pd.DataFrame(MSE)
    MSE.index = [i for i in range(1, 46)]
    # print(MSE)
    print('MSE SARIMA-MODEL: ', MSE.mean())


# FB PROPHET STORE SALES FORECAST

def predict_all_stores_prophet():

    predicted_stores_prophet = pd.DataFrame(columns=range(0, len(stores_train_prophet.columns)))
    for store in range(1, len(stores_train_prophet.columns)):
        m = Prophet(yearly_seasonality=15)
        m.fit(stores_train_prophet.iloc[:, [0, store]])
        future = m.make_future_dataframe(
            freq='W', periods=len(stores_df)-len(stores_train_prophet), include_history=False)
        forecast = m.predict(future)
        predicted_stores_prophet[store] = forecast['yhat']

    predicted_stores_prophet.to_csv('predicted_stores_prophet.csv')
    return predicted_stores_prophet
    print('saved')


# read in predicted data from csv and assigt new index (saved above)
predictions_prophet = pd.read_csv('predicted_stores_prophet.csv', sep=',', index_col=0)
predictions_prophet.drop(columns='0', inplace=True)
predictions_prophet.index.name = 'Date'
predictions_prophet.index = stores_test.index


def plot_predictions_vs_actual_prophet():  # plotterinho

    for i in range(len(predictions_prophet.columns)):
        plt.plot(predictions_prophet.iloc[:, i], c='r')
        plt.plot(stores_test.iloc[:, i], c='b')
        plt.show()


def mse_prophet():  # mean square errors

    MSE = []
    for store in range(0, len(predictions_prophet.columns)):
        error = mean_squared_error(
            stores_test.iloc[:, store], predictions_prophet.iloc[:, store])
        MSE.append(error)
        # print(store+1, ' MSE: ', error)
    MSE = pd.DataFrame(MSE)
    MSE.index = [i for i in range(1, 46)]
    # print(MSE)
    print('MSE PROPHET-MODEL: ', MSE.mean())


# HWES STORE SALES FORECAST
# create and fit model
def predict_all_stores_hwes():

    predicted_stores_hwes = pd.DataFrame(columns=range(1, len(stores_train.columns)))
    for store in range(0, len(stores_train.columns)):
        hwes_model = ExponentialSmoothing(
            stores_train.iloc[:, store], seasonal='add', seasonal_periods=52)
        hwes_fit = hwes_model.fit()
        hwes_yhat = hwes_fit.predict(len(stores_train), len(stores_df)-1)
        predicted_stores_hwes[store+1] = hwes_yhat
        print(predicted_stores_hwes[store+1])

    predicted_stores_hwes.to_csv('predictions_hwes.csv')


predictions_hwes = pd.read_csv('predictions_hwes.csv', sep=',', index_col=0)
predictions_hwes.index = stores_test.index


def mse_hwes():  # mean square errors

    MSE = []
    for store in range(0, len(predictions_hwes.columns)):
        error = mean_squared_error(
            stores_test.iloc[:, store], predictions_hwes.iloc[:, store])
        MSE.append(error)
        # print(store+1, ' MSE: ', error)
    MSE = pd.DataFrame(MSE)
    MSE.index = [i for i in range(1, 46)]
    # print(MSE)
    print('MSE HWES-MODEL: ', MSE.mean())


# PLOT ALL MODELS VS ACTUAL
def plot_predictions_all_models():  # plotterinho

    for i in range(len(predictions_prophet.columns)):
        plt.plot(predictions.iloc[:, i], c='g', label='SARIMA')
        plt.plot(predictions_prophet.iloc[:, i], c='r', label='PROPHET')
        plt.plot(predictions_hwes.iloc[:, i], c='y', label='HWES')
        plt.plot(stores_df.iloc[:, i], c='b', label='ACTUAL')
        plt.axvline(x=stores_df.index[train_length],
                    linestyle='--', color='r', label='forecast cutoff')
        plt.legend()
        wm = plt.get_current_fig_manager()
        wm.window.state('zoomed')
        plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
import os
os.chdir('D:/PythonProjektATOM/Git/Repositories/Time-series-forecasting/Walmart sales/')

prediction_df = pd.read_csv('prediction_df.csv', sep=',', header=None, index_col=0)

prediction_df.index.name = 'ds'
prediction_df.columns = ['y']
prediction_df.index = pd.to_datetime(prediction_df.index, format="%Y/%m/%d")

# test & train
train_df = prediction_df.iloc[:80]
test_df = prediction_df.iloc[80:len(prediction_df)]

print(train_df.head())
print(test_df.shape)

train_series = train_df
test_series = test_df


def plot_acf_pacf():  # ACF and PACF residual plots
    plot_acf(prediction_df, lags=52)
    plot_pacf(prediction_df, lags=52)
    plt.show()


# fit sarimax model
model = SARIMAX(train_series, order=(0, 1, 0), seasonal_order=(1, 0, 0, 52))
model_fit = model.fit(disp=False)
yhat = model_fit.predict(len(train_series), len(prediction_df)-1, typ='levels')
error = mean_squared_error(test_series, yhat)
print('Test MSE: %.3f' % error)


def plot_result():  # out of sample result
    plt.plot(test_series)
    plt.plot(yhat, color='red')
    plt.show()


# FB PROPHET FORECAST
train_series = train_series.reset_index()

# fit prophet model
m = Prophet(yearly_seasonality=30)
m.fit(train_series)

# make prediction dataframe
future = m.make_future_dataframe(freq='W', periods=len(prediction_df)-80)
forecast = m.predict(future)


def prophet_results():  # plot results
    fig1 = m.plot(forecast)
    plt.plot(test_series, c='black', alpha=0.5)
    plt.axvline(x=train_series.iloc[-1, 0], c='r')
    plt.show()

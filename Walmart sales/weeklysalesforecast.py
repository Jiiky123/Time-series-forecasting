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

# importing prediction_df whick includes aggregated sales
prediction_df = pd.read_csv('prediction_df.csv', sep=',', header=None, index_col=0)

prediction_df.index.name = 'ds'
prediction_df.columns = ['y']
prediction_df.index = pd.to_datetime(prediction_df.index, format="%Y/%m/%d")


# SARIMAX AGG. SALES FORECAST -----------------------------------------------

# test & train
train_df = prediction_df.iloc[:80]
test_df = prediction_df.iloc[80:len(prediction_df)]

train_series = train_df
test_series = test_df


def plot_acf_pacf():  # ACF and PACF residual plots
    plot_acf(prediction_df, lags=52)
    plot_pacf(prediction_df, lags=52)
    plt.show()


def SARIMAX_fit():  # fit sarimax model
    model = SARIMAX(train_series, order=(0, 1, 0), seasonal_order=(1, 0, 0, 52))
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(train_series), len(prediction_df)-1, typ='levels')
    error = mean_squared_error(test_series, yhat)
    print('Test MSE: %.3f' % error)


def plot_result():  # out of sample result
    plt.plot(test_series)
    plt.plot(yhat, color='red')
    plt.show()
# -----------------------------------------------------------------------------


# FB PROPHET AGG. SALES FORECAST -----------------------------------------------
def Prophet_fit():
    train_series = train_series.reset_index()

    # fit prophet model
    m = Prophet(yearly_seasonality=30)
    m.fit(train_series)

    # make prediction dataframe
    future = m.make_future_dataframe(freq='W', periods=len(prediction_df)-80)
    forecast = m.predict(future)

    # plot results
    fig1 = m.plot(forecast)
    plt.plot(test_series, c='black', alpha=0.5)
    plt.axvline(x=train_series.iloc[-1, 0], c='r')
    plt.show()

# ----------------------------------------------------------------------------

# SALES BY STORE AND DEPARTMENT -------------------------------------------


# importing merged_feat_train dataset which includes store/dept sales
merged_feat_train = pd.read_csv('merged_feat_train.csv', sep=',', index_col=0)


# make a df with store as column and weekly sales as rows
stores_df = merged_feat_train[['Store', 'Dept', 'Weekly_Sales']]
stores_df = stores_df.pivot_table(values='Weekly_Sales', index=stores_df.index, columns='Store')
stores_df.index = pd.to_datetime(stores_df.index, format="%Y/%m/%d")


stores_train = stores_df.iloc[:102]
stores_train.index = pd.DatetimeIndex(stores_train.index.values,
                                      freq=stores_train.index.inferred_freq)
stores_test = stores_df.iloc[102:len(stores_df)-1]


def plot_stores():
    stores_df.plot()
    plt.show()


def predict_all_stores():
    predicted_stores = pd.DataFrame(columns=range(1, 45))
    for store in range(len(stores_df.columns)):
        model = SARIMAX(stores_train.iloc[:, store], order=(0, 1, 0), seasonal_order=(1, 0, 0, 52))
        model_fit = model.fit(disp=False)
        yhat = pd.DataFrame(model_fit.predict(len(stores_train), len(stores_df)-1, typ='levels'))
        predicted_stores[store] = yhat[0]
    predicted_stores.to_csv('predicted_stores.csv')
    print('saved')


# predict_all_stores()

predictions = pd.read_csv('predicted_stores.csv', sep=',', index_col=0)
predictions = predictions.rename(columns={'0': '45'})
predictions.index = pd.to_datetime(predictions.index, format="%Y/%m/%d")

# optimistic / pessimistic predictions
predictions_best_perf = pd.Series([(predictions.iloc[-1, store]-predictions.iloc[0, store]) /
                                   predictions.iloc[0, store] for store in range(len(predictions.columns))])
predictions_best_perf.index = np.arange(1, len(predictions_best_perf)+1)
print(predictions_best_perf.sort_values())

# predictions.plot()
# plt.tight_layout()
# plt.show()

# predictions.iloc[:, 10].plot()
# # stores_df.iloc[:, 30].plot()
# stores_test.iloc[:, 11].plot()
# plt.show()


def plot_predictions():
    for i in range(len(predictions)):
        plt.plot(predictions.iloc[:, i], c='r')
        plt.plot(stores_test.iloc[:, i+1], c='b')
        plt.show()


plot_predictions()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import scatter_matrix
os.chdir('D:/PythonProjektATOM/Git/Repositories/Time-series-forecasting/Walmart sales/')

# import csv's
features_df = pd.read_csv('features.csv', sep=',')
train_df = pd.read_csv('train.csv', sep=',')
test_df = pd.read_csv('test.csv', sep=',')
stores_df = pd.read_csv('stores.csv', sep=',')

# drop columns
features_df = features_df.drop(columns=['Fuel_Price', 'CPI'])


def df_info():  # inspect df's
    print(features_df.info())
    print(train_df.info())
    print(test_df.info())
    print(stores_df.info())


def df_desc():  # describe
    print(features_df.describe())
    print(train_df.describe())
    print(test_df.describe())
    print(stores_df.describe())


def df_shape():  # df shapes
    print(features_df.shape)
    print(train_df.shape)
    print(test_df.shape)
    print(stores_df.shape)


def df_na():  # NA-values
    print(features_df.isna().sum())
    print(train_df.isna().sum())
    print(test_df.isna().sum())
    print(stores_df.isna().sum())


# convert df's to date-time
features_df['Date'] = pd.to_datetime(features_df['Date'], format="%Y/%m/%d")
train_df['Date'] = pd.to_datetime(train_df['Date'], format="%Y/%m/%d")
test_df['Date'] = pd.to_datetime(test_df['Date'], format="%Y/%m/%d")

# set date as index
features_df = features_df.set_index('Date')
train_df = train_df.set_index('Date')
test_df = test_df.set_index('Date')

features_df.sort_values('Date', inplace=True)
train_df.sort_values('Date', inplace=True)
test_df.sort_values('Date', inplace=True)

# create train test df for prediction
prediction_df = train_df.groupby('Date').sum().sort_values('Date')
prediction_df = prediction_df['Weekly_Sales']

print(prediction_df.head())
print(features_df.shape)
print(train_df.shape)
print(stores_df.shape)

prediction_df.to_csv('prediction_df.csv')


# def store_sales_plot(start, end):  # weekly sales by store
#     for store in range(start, end):
#         store_sales = train_df[train_df['Store'] == store]
#         store_sales = store_sales.resample('W-MON').sum()
#         plt.plot(store_sales.index, store_sales.Weekly_Sales)
#
#     plt.show()
#
#
# def seasonal_plot():  # seasonal plot
#     sales_seasonal = seasonal_decompose(all_store_sales.Weekly_Sales, freq=4)
#     sales_seasonal.plot()
#     plt.show()

# scatter_matrix(features_df.iloc[:, 2:9], alpha=0.2, figsize=(12, 8), diagonal='kde')
# plt.tight_layout()
# plt.show()

# combine train_df to features_df
# train_df_new = features_df.join(train_df['Weekly_Sales'], how='inner')
#
# print(train_df_new.shape)

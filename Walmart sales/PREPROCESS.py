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

# merged dataset on date and store columns
merged_feat_train = pd.merge(train_df, features_df, on=['Date', 'Store'])
merged_feat_train = merged_feat_train.drop(columns=['IsHoliday_x', 'IsHoliday_y',
                                                    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'])
merged_feat_train = merged_feat_train.iloc[:50000]


# set date as index
merged_feat_train = merged_feat_train.set_index('Date')
features_df = features_df.set_index('Date')
train_df = train_df.set_index('Date')
test_df = test_df.set_index('Date')

merged_feat_train.sort_values(['Date', 'Store'], inplace=True)
features_df.sort_values('Date', inplace=True)
train_df.sort_values('Date', inplace=True)
test_df.sort_values('Date', inplace=True)

# create train test df for prediction
prediction_df = train_df.groupby('Date').sum().sort_values('Date')
prediction_df = prediction_df['Weekly_Sales']


def print_shape():
    print(merged_feat_train.shape)
    print(prediction_df.shape)
    print(features_df.shape)
    print(train_df.shape)
    print(stores_df.shape)


def quick_analysis():  # quick dirty visual analysis
    scatter_matrix(merged_feat_train, alpha=0.2, figsize=(12, 8), diagonal='kde')
    plt.tight_layout()
    plt.show()


def plot_hist():
    merged_feat_train.hist()
    plt.show()


print(merged_feat_train.info())
plot_hist()
# merged_feat_train.to_csv('merged_feat_train.csv')


# def store_sales_plot(start, end):  # weekly sales by store
#     for store in range(start, end):
#         store_sales = train_df[train_df['Store'] == store]
#         store_sales = store_sales.resample('W-MON').sum()
#         plt.plot(store_sales.index, store_sales.Weekly_Sales)
#
#     plt.show()

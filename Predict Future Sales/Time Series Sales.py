#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import Holt
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from math import sqrt

from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('data/sales_train.csv')
df


# We gona change 'date' feature to other format. It's good practice beacause our new format will be used by many functions in the future

# In[3]:


# Change format of date feature
df['date'] = pd.to_datetime(df.date, format = '%d.%m.%Y')
df.head().style.set_properties(subset=['date'], **{
    'background-color': 'dodgerblue'
})


# In[5]:


# Missing values
df.isnull().sum()


# We can see that there is no missing values

# # Resampling

# In[6]:


plot = df.groupby('date')['shop_id'].count()

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=plot.index, y=plot.values)
)

fig.update_layout(
    title='Number of sold products every day',
    yaxis_title='Number of products',
    xaxis_title='Month'
)


# In[7]:


plot = df.groupby('date', as_index=False)['shop_id'].count()
resampled_plot = plot[['date', 'shop_id']].resample('7D', on='date').sum().reset_index(drop=False)

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=resampled_plot.date, y=resampled_plot.shop_id)
)

fig.update_layout(
    title='Number of sold products every week',
    yaxis_title='Number of products',
    xaxis_title='Month'
)


# In[8]:


plot = df.groupby('date_block_num')['shop_id'].count()

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=plot.index, y=plot.values)
)

fig.update_layout(
    title='Number of sold products every month',
    yaxis_title='Number of products',
    xaxis_title='Month'
)


# There is no necessity to look at the daily data. Considering weekly data seems to be sufficient as well. We gona create new dataframe with number of sold_products ordered by date and we will downsample it

# In[9]:


df_num_prod = pd.DataFrame(df.groupby('date', as_index=False)['shop_id'].count()).rename(columns={'shop_id': 'sold_products'})
df_downsampled = df_num_prod[[
    'date',
    'sold_products'
]].resample('7D', on='date').sum().reset_index(drop=False)
df_downsampled


# # Autocorrelation

# In[10]:


fig = go.Figure()

for i in range(2):
    fig.add_trace(
        go.Scatter(x=df_downsampled.date, y=df_downsampled['sold_products'].shift(-i), name='lag ' + str(i))
    )
    
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Sold Products')
fig.update_layout(title='Lag plot')
    
fig.show()


# In[11]:


# Running this function plots the temperature data (t) on the x-axis
autocorrelation_plot(df_downsampled['sold_products'])


# In[12]:


plot_acf(df_downsampled['sold_products'])


# In[13]:


plot_pacf(df_downsampled['sold_products'])


# In[14]:


# Running this function plots the temperature data (t) on the x-axis against the temperature on the previous data 
# (t-1) on the y-axis
lag_plot(df_downsampled['sold_products'])


# When data have a trend, the autocorrelations for small lags tend to be large and positive because observations nearby in time are also nearby in size. So the ACF of trended time series tend to have positive values that slowly decrease as the lags increase.
# When data are seasonal, the autocorrelations will be larger for the seasonal lags (at multiples of the seasonal frequency) than for other lags.
# When data are both trended and seasonal, you see a combination of these effects.
# In our case we can see strong trend.

# For white noise series, we expect each autocorrelation to be cose to zero. Of course, they will not be exactly equal to zero as there is some random variation. For a white noise series, we expect 95% of the spikes in the ACF to lie within plus-minus 2/sqrt(T) where T is the lenght of the time series. In our case autocorrelation coefficients are outside of the range that is why our data are not white noise.

# #  Simple forecasting methods

# Average Method 

# In[15]:


# Here, the forecasts of all future values are equal to the average
rang = 20
mean = df_downsampled['sold_products'].mean()
means = []
for i in range(rang):
    means.append(mean)
    
new = pd.date_range(df_downsampled.date.iloc[-1], periods=rang, freq='W')
new_df = pd.DataFrame({'date': new[1:], 'sold_products': mean})
copy_df = df_downsampled.copy()

to_plot = pd.concat([copy_df, new_df])

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=to_plot.date, y=to_plot.sold_products, name='basic')
)

fig.add_trace(
    go.Scatter(x=new_df.date, y=new_df.sold_products, name='predicted', mode='lines')
)


# Naïve method

# In[16]:


# For naïve forecasts, we simply set all forecasts to be the value of the last observation
rang = 20
mean = df_downsampled['sold_products'].iloc[-1]
means = []
for i in range(rang):
    means.append(mean)
    
new = pd.date_range(df_downsampled.date.iloc[-1], periods=rang, freq='W')
new_df = pd.DataFrame({'date': new[1:], 'sold_products': mean})
copy_df = df_downsampled.copy()

to_plot = pd.concat([copy_df, new_df])

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=to_plot.date, y=to_plot.sold_products, name='basic')
)

fig.add_trace(
    go.Scatter(x=new_df.date, y=new_df.sold_products, name='predicted', mode='lines')
)


# # Decomposition

# In[17]:


decomp = seasonal_decompose(df_downsampled.sold_products, freq=10, model='additive', extrapolate_trend='freq')
df_downsampled['sold_products_trend'] = decomp.trend
df_downsampled['sold_products_seasonal'] = decomp.seasonal
df_downsampled['sold_products_residual'] = decomp.resid


# In[18]:


fig = make_subplots(cols=1, rows=4, subplot_titles=(
    'Basic',
    'Trend',
    'Seasonality',
    'Residual'
))

fig.add_trace(
    go.Scatter(x=df_downsampled.date, y=df_downsampled.sold_products),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(x=df_downsampled.date, y=df_downsampled.sold_products_trend),
    row=2,
    col=1
)

fig.add_trace(
    go.Scatter(x=df_downsampled.date, y=df_downsampled.sold_products_seasonal),
    row=3,
    col=1
)

fig.add_trace(
    go.Scatter(x=df_downsampled.date, y=df_downsampled.sold_products_residual),
    row=4,
    col=1
)

fig.update_layout(height=800, title_text='Eecomposition of Sold Products', showlegend=False)


# #  Stationarity

# Plot time series and check for trends or seasonality

# In[19]:


fig = go.Figure()

fig.add_trace(
    go.Scatter(x=df_downsampled.date, y=df_downsampled.sold_products, name='basic')
)

fig.add_trace(
    go.Scatter(x=df_downsampled.date, y=df_downsampled.sold_products.rolling(10).mean(), name='rolling mean')
)

fig.add_trace(
    go.Scatter(x=df_downsampled.date, y=df_downsampled.sold_products.rolling(10).std(), name='rolling std')
)


# ADF test is used to determine the presense of unit root in the series, and hence helps in understanding if the series is stationary or not. The null and alternate hypothesis of this test are:
# Null Hypothesis: The series has a unit root
# Alternate Hypothesis: The series has no unit root

# In[20]:


def adf_test(timeseries):
    print('Results of Diskey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# KPSS is another test for checking the stationarity of a time series. The null and alternate hypothesis for the KPSS test are opposite that of the ADF test:
# Null Hypothesis: The process is trend stationary
# Alternate Hypothesis: The series has a unit root (series is not stationary)

# In[21]:


def kpss_test(timeseries):
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags='auto')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print(kpss_output)


# In[22]:


adf_test(df_downsampled.sold_products)


# If the test statistic is less than the critical value, we can reject the null hypothesis (aka the series is stationary). When the test statistic is greater than the critical value, we fail to reject the null hypothesis (which means the series is not stationary). In out case, the test statistic > critical value, which implies that the series is not stationary.

# In[23]:


kpss_test(df_downsampled.sold_products)


# If the test statistic is greater than the critical value, we reject the null hypothesis (series is not stationary). If the test statistic is less than the critical value, it fail to reject the null hypothesis (series is stationary). In our case, the test statistic > critical value, which again implies that the series is not stationary.

# In both tests we see that our data are not stationary. That is why we will use differencing to make it stationary for later use

# In[24]:


# First Order Differencing
ts_diff = np.diff(df_downsampled.sold_products)
df_downsampled['sold_products_diff_1'] = np.append([0], ts_diff)

# Second Order Differencing
ts_diff = np.diff(df_downsampled.sold_products_diff_1)
df_downsampled['sold_products_diff_2'] = np.append([0], ts_diff)


# In[25]:


fig = make_subplots(cols=1, rows=2, subplot_titles=(
    'sold_products_diff_1',
    'sold_products_diff_2'
))

fig.add_trace(
    go.Scatter(x=df_downsampled.date, y=df_downsampled.sold_products_diff_1, legendgroup='basic', line=dict(color = 'blue'), name='basic'),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(x=df_downsampled.date, y=df_downsampled.sold_products_diff_1.rolling(10).mean(), legendgroup='mean', line=dict(color = 'orange'), name='rolling mean'),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(x=df_downsampled.date, y=df_downsampled.sold_products_diff_1.rolling(10).std(), legendgroup='std', line=dict(color = 'red'), name='rolling std'),
    row=1,
    col=1
)
fig.update_yaxes(title_text='Number of Sold Products', row=1, col=1)


fig.add_trace(
    go.Scatter(x=df_downsampled.date, y=df_downsampled.sold_products_diff_2, legendgroup='basic', line=dict(color = 'blue'), showlegend=False),
    row=2,
    col=1
)
fig.add_trace(
    go.Scatter(x=df_downsampled.date, y=df_downsampled.sold_products_diff_2.rolling(10).mean(), legendgroup='mean', line=dict(color = 'orange'), showlegend=False),
    row=2,
    col=1
)
fig.add_trace(
    go.Scatter(x=df_downsampled.date, y=df_downsampled.sold_products_diff_2.rolling(10).std(), legendgroup='std', line=dict(color = 'red'), showlegend=False),
    row=2,
    col=1
)
fig.update_yaxes(title_text='Number of Sold Products', row=2, col=1)

fig.update_layout(showlegend=False)


# In[26]:


adf_test(df_downsampled.sold_products_diff_1)


# After differencing we can see that, the ADF test statistic < critical value, which implies that the series is now stationary.

# In[27]:


kpss_test(df_downsampled.sold_products_diff_1)


# After differencing we can see that, the KPSS test statistic < critical value, which implies that the series is now stationary.

# In[28]:


plot_acf(df_downsampled.sold_products_diff_1)


# In[29]:


plot_pacf(df_downsampled.sold_products_diff_1)


# # Autoregression Models

# In[30]:


# Split dataset
X = df_downsampled.copy()
X = X.drop(['date'], axis=1)
X = X.sold_products_diff_1.values
train, test = X[1:len(X)-7], X[len(X)-7:]

# Train autoregression
model = AutoReg(train, lags=2)
model_fit = model.fit()
print('Coefficients %s' % model_fit.params)
print()

# Make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))

print()
    
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# Plot results
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=np.arange(7), y=test, name='test')
)

fig.add_trace(
    go.Scatter(x=np.arange(7), y=predictions, name='predictions')
)


# # ARIMA Model

# In[31]:


model = ARIMA(df_downsampled.set_index('date').sold_products, order=(5, 1, 0))
model_fit = model.fit()
print(model_fit.summary())


# In[32]:


X = df_downsampled.set_index('date').sold_products
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

# walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(2,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

print()
# Evaluate forecast
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# Plot forecast against actual outcomes
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=np.arange(30), y=test, name='test')
)

fig.add_trace(
    go.Scatter(x=np.arange(30), y=predictions, name='predictions')
)


# In[33]:


def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.8)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return sqrt(error)

# GriSearchCV for ARIMA
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float('inf'), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order, rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


# # Simple Exponential Smoothing

# The simplest of the exponentially smoothing methods is naturally called simple exponencial smoothing (SES). This method is suitable for forecasting data with no clear trend or seasonal pattern. Here we gona use it only to show how it is works.

# In[34]:


X = df_downsampled.set_index('date').sold_products
fit1 = SimpleExpSmoothing(X, initialization_method='heuristic').fit(smoothing_level=0.2, optimized=False)
fcast1 = fit1.forecast(20)


fit2 = SimpleExpSmoothing(X, initialization_method='heuristic').fit(smoothing_level=0.4, optimized=False)
fcast2 = fit1.forecast(20)

fit3 = SimpleExpSmoothing(X, initialization_method='heuristic').fit(smoothing_level=0.6, optimized=False)
fcast3 = fit1.forecast(20)


# In[35]:


fig = go.Figure()

fig.add_trace(
    go.Scatter(x=X.index, y=X.values, name='test')
)

fig.add_trace(
    go.Scatter(x=fit1.fittedvalues.index, y=fit1.fittedvalues.values, legendgroup='0.2', name='SES 0.2', line=dict(color = 'red'))
)

fig.add_trace(
    go.Scatter(x=fcast1.index, y=fcast1.values, legendgroup='0.2', showlegend=False, line=dict(color = 'red'))
)

fig.add_trace(
    go.Scatter(x=fit2.fittedvalues.index, y=fit2.fittedvalues.values, legendgroup='0.4', name='SES 0.4', line=dict(color = 'yellow'))
)

fig.add_trace(
    go.Scatter(x=fcast2.index, y=fcast2.values, legendgroup='0.4', showlegend=False, line=dict(color = 'yellow'))
)

fig.add_trace(
    go.Scatter(x=fit3.fittedvalues.index, y=fit3.fittedvalues.values, legendgroup='0.6', name='SES 0.6', line=dict(color='brown'))
)

fig.add_trace(
    go.Scatter(x=fcast3.index, y=fcast3.values, legendgroup='0.6', showlegend=False, line=dict(color = 'brown'))
)


# In[36]:


X = df_downsampled.set_index('date').sold_products
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

# walk-forward validation
for t in range(len(test)):
    model = SimpleExpSmoothing(history, initialization_method='heuristic').fit(smoothing_level=0.2, optimized=False)
#     model_fit = model.fit()
    output = model.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

print()
# Evaluate forecast
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# Plot forecast against actual outcomes
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=np.arange(30), y=test, name='test')
)

fig.add_trace(
    go.Scatter(x=np.arange(30), y=predictions, name='predictions')
)


# # Holt's Exponential Smoothing

# In[37]:


X = df_downsampled.set_index('date').sold_products
fit1 = Holt(X, initialization_method='estimated').fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
fcast1 = fit1.forecast(10)

fit2 = Holt(X, initialization_method='estimated').fit(smoothing_level=0.6, smoothing_trend=0.4, optimized=False)
fcast2 = fit2.forecast(10)

fit3 = Holt(X, initialization_method='estimated').fit(smoothing_level=0.4, smoothing_trend=0.6, optimized=False)
fcast3 = fit3.forecast(10)


# In[38]:


fig = go.Figure()

fig.add_trace(
    go.Scatter(x=X.index, y=X.values, name='test')
)

fig.add_trace(
    go.Scatter(x=fit1.fittedvalues.index, y=fit1.fittedvalues.values, legendgroup='HES 0.8 0.2', line=dict(color = 'red'), name='HES 0.8 0.2')
)
fig.add_trace(
    go.Scatter(x=fcast1.index, y=fcast1.values, mode='lines', legendgroup='HES 0.8 0.2', line=dict(color='red'), showlegend=False)
)

fig.add_trace(
    go.Scatter(x=fit2.fittedvalues.index, y=fit2.fittedvalues.values, legendgroup='HES 0.6 0.4', line=dict(color = 'yellow'), name='HES 0.6 0.4')
)
fig.add_trace(
    go.Scatter(x=fcast2.index, y=fcast2.values, mode='lines', legendgroup='HES 0.6 0.4', line=dict(color='yellow'), showlegend=False)
)

fig.add_trace(
    go.Scatter(x=fit3.fittedvalues.index, y=fit3.fittedvalues.values, legendgroup='HES 0.4 0.6', line=dict(color = 'brown'), name='HES 0.4 0.6')
)
fig.add_trace(
    go.Scatter(x=fcast3.index, y=fcast3.values, mode='lines', legendgroup='HES 0.4 0.6', line=dict(color='brown'), showlegend=False)
)


# In[39]:


X = df_downsampled.set_index('date').sold_products
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

# walk-forward validation
for t in range(len(test)):
    model = Holt(history, initialization_method='estimated').fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
    output = model.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

print()
# Evaluate forecast
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# Plot forecast against actual outcomes
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=np.arange(30), y=test, name='test')
)

fig.add_trace(
    go.Scatter(x=np.arange(30), y=predictions, name='predictions')
)


# In[40]:


simulated = model_fit.simulate(anchor='end', nsimulations=7, repetitions=100)


# In[41]:


fig = go.Figure()

for i in range(len(simulated)):
    fig.add_trace(
        go.Scatter(x=np.arange(20), y=simulated[i], line=dict(color = 'gray'), showlegend=False, opacity=0.2)
    )
    
fig.add_trace(
    go.Scatter(x=np.arange(20), y=test, line=dict(color = 'red'))
)
fig.show()


# # Holt's Winters Seasonal Smoothing

# In[42]:


X = df_downsampled.set_index('date').sold_products

fit1 = ExponentialSmoothing(X, seasonal_periods=12, trend='add', seasonal='add', use_boxcox=True, initialization_method='estimated').fit()
fcast1 = fit1.forecast(20)


# In[43]:


fig = go.Figure()

fig.add_trace(
    go.Scatter(x=X.index, y=X.values, name='data')
)

fig.add_trace(
    go.Scatter(x=fit1.fittedvalues.index, y=fit1.fittedvalues.values, legendgroup='add add', line=dict(color = 'red'), name='predicted')
)

fig.add_trace(
    go.Scatter(x=fcast1.index, y=fcast1.values, legendgroup='add add', line=dict(color = 'red'), showlegend=False)
)


# In[44]:


X = df_downsampled.set_index('date').sold_products
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

# walk-forward validation
for t in range(len(test)):
    model = ExponentialSmoothing(history, seasonal_periods=7, trend='mul', seasonal='mul', use_boxcox=True, initialization_method='estimated')
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

print()
# Evaluate forecast
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# Plot forecast against actual outcomes
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=np.arange(30), y=test, name='test')
)

fig.add_trace(
    go.Scatter(x=np.arange(30), y=predictions, name='predictions')
)


# In[45]:


simulated = model_fit.simulate(anchor='end', nsimulations=7, repetitions=100)


# In[46]:


simulated[0]


# In[47]:


fig = go.Figure()

for i in range(len(simulated)):
    fig.add_trace(
        go.Scatter(x=np.arange(20), y=simulated[i], line=dict(color = 'gray'), showlegend=False, opacity=0.2)
    )
    
fig.add_trace(
    go.Scatter(x=np.arange(20), y=test, line=dict(color = 'red'))
)

fig.show()


# In[ ]:





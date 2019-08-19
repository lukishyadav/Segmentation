
import my_module
import pandas as pd



from bokeh.io import curdoc
import logging
from bokeh.layouts import column,layout,row,widgetbox
import pandas as pd
#import my_module
import datetime
import seaborn as sns
from pyproj import Proj
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.tile_providers import CARTODBPOSITRON ,CARTODBPOSITRON_RETINA
import numpy as np
from sklearn.cluster import DBSCAN 
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider,TextInput
from bokeh.palettes import Category20


"""
df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/Darwin_Demand.csv')
from datetime import datetime
df['Day']=df['date'].apply(lambda x:datetime.strptime(x[0:19],'%Y-%m-%d %H:%M:%S'))
#DF['Date']=DF['Date'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
fd=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/supply_demand/last7days.csv')
fd['Day']=fd['time'].apply(lambda x:datetime.strptime(x[0:19],'%Y-%m-%d %H:%M'))
DF=pd.merge(df,fd,on='Day',how='inner')
FD=DF.groupby(['day']).sum(name='Counts')
"""

#df=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/supply_demand/supply_demand_counts_20190501_20190606.csv')
df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/Darwin_Demand.csv')
DF=df.tail(1300).copy()
DF=df.copy()
#plt.plot(DF['Rental_Count'])
import datetime
DF['day']=DF['date'].apply(lambda x:datetime.datetime.strptime(x[0:10],'%Y-%m-%d'))
FD=DF.groupby(['day']).sum(name='Counts')
data=FD['counts']
data=data.to_frame()

data=FD['counts']
data=data.to_frame()

#import pandas
#import matplotlib.pyplot as plt

#plt.plot(DF['Supply_Count'])

['Unnamed: 0', 'Date', 'Appopen_Count', 'Supply_Count', 'Rental_Count',
       'hour']


data=DF['counts']

data=FD['counts']


from pandas import DataFrame
from pandas import concat
from pandas import read_csv


# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	raw_values = raw_values.reshape(len(raw_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(raw_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return train, test


#train, test = prepare_data(series, n_test, n_lag, n_seq)

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	# transform data to be stationary
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(raw_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return scaler, train, test



def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = test[:,(n_lag+i)]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))
        



# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i
		off_e = off_s + len(forecasts[i])
		xaxis = [x for x in range(off_s, off_e)]
		pyplot.plot(xaxis, forecasts[i], color='red')
	# show the plot
	pyplot.show()


# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
	# reshape training into [samples, timesteps, features]
	X, y = train[:, 0:n_lag], train[:, n_lag:]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
		model.reset_states()
	return model  

# fit an LSTM network to training data
def fit_lstm2(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
	# reshape training into [samples, timesteps, features]
	X, y = train[:, 0:n_lag], train[:, n_lag:]
	X = X.reshape(X.shape[0],X.shape[1],1)
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
		model.reset_states()
	return model  


# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]

# make one forecast with an LSTM,
def forecast_lstm2(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1,len(X),1)
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]

# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts

# evaluate the persistence model
def make_forecasts2(model, n_batch, train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = forecast_lstm2(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts

"""

Predict Unseen

"""


# evaluate the persistence model
def make_unseen_forecasts(model, n_batch, series, n_lag, n_seq):
	forecasts = list()
	#for i in range(len(test)):
	#X, y = test[-1, 0:n_lag], test[-1, n_lag:]
	X=series.values[-3:]
    
	X=X.reshape(1,X.shape[0],X.shape[1])
	forecast = model.predict(X, batch_size=n_batch)
	#forecasts.append(forecast)
	return forecast



#Test=actual


def evaluate_forecasts(Test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in Test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))



def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]

		inverted.append(inv_scale)
	return inverted
# =============================================================================
#     
# data=DF['counts'].to_frame()
# supervised = series_to_supervised(data, 1, 3)
# 
# n_lag = 2
# n_seq = 3
# n_test = 10
# train, test = prepare_data(data, n_test, n_lag, n_seq)
# =============================================================================

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array




# =============================================================================
# # fit model
# model = fit_lstm(train, 1, 3, 1, 500, 1)
# 
# # make forecasts
# forecasts = make_forecasts(model, 1, train, test, 1, 3)
# 
# actual = [row[n_lag:] for row in test]
# 
# evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# 
# %matplotlib auto
# plot_forecasts(data, forecasts, n_test+2)
# =============================================================================


series=FD['counts'].to_frame()

n_lag = 3
n_seq = 7
n_test = 10
n_epochs = 50
n_batch = 1
n_neurons = 20
# prepare data
scaler,train, test = prepare_data(series, n_test, n_lag, n_seq)
# fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)

model = fit_lstm2(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)


print(model.summary())


# make forecasts
forecasts = make_forecasts2(model, n_batch, train, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2)
# evaluate forecasts
evaluate_forecasts(actual, forecasts,s n_lag, n_seq)
# plot forecasts
%matplotlib auto
plot_forecasts(series, forecasts, n_test+2)


predictions=make_unseen_forecasts(model, n_batch, series, n_lag, n_seq)

outputs=scaler.inverse_transform(predictions)
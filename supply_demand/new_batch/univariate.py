#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:31:19 2019

@author: lukishyadav
"""

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


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


from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot
from keras.models import load_model

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
import numpy



#df=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/supply_demand/supply_demand_counts_20190501_20190606.csv')
df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/Darwin_Demand.csv')


DF=df.tail(1300).copy()

DF=df.copy()

#plt.plot(DF['Rental_Count'])
import datetime

DF['day']=DF['date'].apply(lambda x:datetime.datetime.strptime(x[0:10],'%Y-%m-%d'))


FD=DF.groupby(['day']).sum(name='Counts')


# =============================================================================
# from matplotlib import pyplot as plt
# plt.plot(FD['counts'])
# 
# =============================================================================


# =============================================================================
#for x in range(5, 0, -1):
#    print(x)
# 
# =============================================================================

#check = [DF['counts'].head(5).shift(i) for i in range(1, 3)]

#DF['date'].iloc[1]


#data=DF['counts']


data=FD['counts']


data=data.to_frame()

#data=diff_values

#data=data.values



# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')


#lag=2
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(lag,0,-1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

def timeseries_to_supervised2(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(lag,0,-1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.dropna(inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model


def fit_lstm2(train,batch_size,nb_epoch,neurons,act,sf=False):
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    X,y=train[:,0:-1],train[:,-1]
    X=X.reshape(X.shape[0],1,X.shape[1])
    model=Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=sf,activation=act))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
    h=model.fit(X, y, epochs=nb_epoch,validation_split=0.2, batch_size=1, verbose=1, shuffle=False,callbacks=[es,mc])
    return model,h,es


def fit_lstm3(train,batch_size,nb_epoch,neurons,act,sf=False):
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    X,y=train[:,0:-1],train[:,-1]
    X=X.reshape(X.shape[0],1,X.shape[1])
    model=Sequential()
    #model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(LSTM(neurons, activation=act,stateful=sf,input_shape=(1,X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
    h=model.fit(X, y, epochs=nb_epoch,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False,callbacks=[es,mc])
    return model,h,es

def fit_lstm4(train,batch_size,nb_epoch,neurons,act,sf=False):
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    X,y=train[:,0:-1],train[:,-1]
    X=X.reshape(X.shape[0],X.shape[1],1)
    model=Sequential()
    #model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(LSTM(neurons, activation=act,stateful=sf,input_shape=(X.shape[1],1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
    h=model.fit(X, y, epochs=nb_epoch,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False,callbacks=[es,mc])
    return model,h,es


def fit_lstm5(train,batch_size,nb_epoch,neurons,act,sf=False,pat=100,lss='mean_squared_error',opt='adam',met=['mse']):
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pat)
    X,y=train[:,0:-1],train[:,-1]
    X=X.reshape(X.shape[0],X.shape[1],1)
    model=Sequential()
    #model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(LSTM(neurons, activation=act,return_sequences=True,stateful=sf,input_shape=(X.shape[1],1)))
    model.add(LSTM(2, activation=act,stateful=sf,input_shape=(X.shape[1],neurons)))
    model.add(Dense(1))
    model.compile(loss=lss, optimizer=opt,metrics=met)
    h=model.fit(X, y, epochs=nb_epoch,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False,callbacks=[es,mc])
    return model,h,es


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# load dataset
#series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

lag=2
series=data

DATA=data.copy()

DATA=DATA.to_frame()

DATA.reset_index(inplace=True)

dates=DATA.iloc[-12:-1,:1].values

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)



#import matplotlib.pyplot as plt
#plt.plot(diff_values)

# transform data to be supervised learning
#supervised = timeseries_to_supervised2(diff_values, lag)
supervised = timeseries_to_supervised2(series, lag)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-12], supervised_values[-12:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
#lstm_model,h,es = fit_lstm5(train_scaled, 1, 1000, 100,'tanh')

lstm_model,h,es = fit_lstm5(train_scaled, 1, 1000, 100,'tanh',False,100,'mean_squared_error','adam',['mse'])

C=h
CC=h.history
#  ['val_loss', 'val_mean_squared_error', 'loss', 'mean_squared_error']

%matplotlib auto
import matplotlib.pyplot as plt

plt.plot(C.history['mean_squared_error'])
plt.plot(C.history['val_mean_squared_error'])
plt.title('model MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


d=h.params

epoch_value=es.stopped_epoch

model = load_model('best_model.h5')
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0:lag].reshape(len(train_scaled), 1, lag)
lstm_model.predict(train_reshaped, batch_size=1)



"""

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[len(train) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
"""

"""
Checksum method
"""



"""
otrain, otest = raw_values[:-12], raw_values[-12:]



X, y = test_scaled[:, 0:-1], test_scaled[:, -1]
X=X.reshape(X.shape[0],1,X.shape[1])
yhat =lstm_model.predict(X,batch_size=1)


TTT=np.append(test_scaled[:, 0:-1],yhat,axis=1)
TTTT=scaler.inverse_transform(TTT)

Yhat=TTTT[:, -1]

cumsum = Yhat.cumsum()
Y = pd.Series(otest[0][0], index=range(len(X)))
Yo=Y.add(cumsum)

predictions=Yo
"""
X, y = test_scaled[:, 0:-1], test_scaled[:, -1]
X=X.reshape(X.shape[0],1,X.shape[1])
X=X.reshape(X.shape[0],X.shape[2],1)
yhat =lstm_model.predict(X,batch_size=1)

TTT=np.append(test_scaled[:, 0:-1],yhat,axis=1)
TTTT=scaler.inverse_transform(TTT)

Yhat=TTTT[:, -1]


#Yt=scaler.inverse_transform(y)

# report performance
rmse = sqrt(mean_squared_error(raw_values[-12:], Yhat))
print('Test RMSE: {0}, Early Stopp Epoch={1}'.format(rmse,epoch_value))
# line plot of observed vs predicted
pyplot.title(label='Test RMSE: %.3f' % rmse)
pyplot.plot(raw_values[-12:],label='Test Data')
pyplot.plot(Yhat,label='Predicted Data')
pyplot.legend()
pyplot.show()



"""

Correct Inverse Differencing

"""


X=[1,2,3,4,5,6]
dv = difference(X, 1)

Y = pd.Series(1, index=range(0,6))
cumsum = dv.cumsum()


y=Y.add(cumsum)

print predictions_ARIMA_diff_cumsum.head()
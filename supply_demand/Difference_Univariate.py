#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:22:17 2019

@author: lukishyadav



Only Check for Test not training!
"""
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


#DF['date'].iloc[1]


#data=DF['counts']


data=FD['counts']


data=data.to_frame()

data=data.values

X=[1,2,3,4,5,6]

"""

Performing Inversion


"""
from pandas import Series

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

global inverse_difference

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

differenced=difference(data,1)


differenced=differenced.to_frame()


"""
# invert transform
inverted = list()
for i in range(len(differenced)):
	value = inverse_difference(data, differenced[i], len(data)-i)
	inverted.append(value)
inverted = Series(inverted)

inverted=inverted.to_frame()
"""




from sklearn.preprocessing import MinMaxScaler
# normalize the dataset
scaler = MinMaxScaler(feature_range=(-1, 1))
Data = scaler.fit_transform(differenced)


dataset=test
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):   #Add +1 to rever changes
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


#data=data.values

# split into train and test sets
train_size = int(len(Data) * 0.67)
test_size = len(Data) - train_size
train, test = Data[0:train_size,:], Data[train_size:len(Data),:]
print(len(train), len(test))


import numpy


nb_epoch=1000


# reshape into X=t and Y=t+1
look_back = 2
n_features=2
n_steps=1

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



# define model
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps,n_features)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
# fit model
#model.fit(trainX, trainY, epochs=200, verbose=2)
h=model.fit(trainX, trainY, epochs=nb_epoch,validation_split=0.2, batch_size=1, verbose=1, shuffle=False,callbacks=[es,mc])


# load the saved model
model = load_model('best_model.h5')
# demonstrate prediction

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



"""

mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)
	X, y = train[:, 0:n_lag+1], train[:, n_lag+1:]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
	# fit network
	h=model.fit(X, y, epochs=nb_epoch,validation_split=0.2, batch_size=n_batch, verbose=1, shuffle=False,callbacks=[es,mc])


"""

import math
from sklearn.metrics import mean_squared_error
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions

trainX = numpy.reshape(trainX, (trainX.shape[0],trainX.shape[2]))
testX = numpy.reshape(testX, (testX.shape[0],testX.shape[2]))


trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


def inv(data,differenced):
    inverted = list()
    for i in range(len(differenced)):
    	value = inverse_difference(data, differenced[i], len(differenced)-i,look_back)
    	inverted.append(value)
    inverted = Series(inverted)
    
    #inverted=inverted.to_frame()
    return inverted


# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


# invert differenced value
def inverse_difference(history, yhat, interval=1,look_back = 1):
	return yhat + history[-interval-look_back+1]

len(testPredict)


testPredict[0]

data[-42:-1]

trainPredict[1][0]

type(trainPredict)

type(trainY)


TrainPredict=inv(data,trainPredict)
#TrainPredict.shape
#TrainPredict=TrainPredict.to_frame().values

TrainY=inv(data,trainY)
TrainY=TrainY[0]



len(testY)

TestPredict = inv(data,testPredict)
#TestPredict=TestPredict.to_frame().values

testY=testY.reshape(-1,1)
TestY = inv(data,testY)

len(testY)

data[-39]

TestY=TestY[0]


#TrainY[0].shape

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(TrainY, TrainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(TestY, TestPredict))
print('Test Score: %.2f RMSE' % (testScore))




#Unscaled
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))






# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(data)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(data)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, :] = testPredict
# plot baseline and predictions

import matplotlib.pyplot as plt
%matplotlib auto
plt.plot(scaler.inverse_transform(data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()



import matplotlib.pyplot as plt
%matplotlib auto
plt.plot(data)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


"""
Train Score: 0.13 RMSE
Test Score: 0.12 RMSE

"""
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
df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/data/Darwin_Demand_19_8.csv')
DF=df.tail(1300).copy()
DF=df.tail(1600).copy()
DF=df.copy()
#plt.plot(df['counts'].tail(160))
import datetime
DF['day']=DF['date'].apply(lambda x:datetime.datetime.strptime(x[0:10],'%Y-%m-%d'))
FD=DF.groupby(['day']).sum(name='Counts')
data=FD['counts']
data=data.to_frame()
data=FD['counts'].iloc[0:-1]

"""
import matplotlib.pyplot as plt
plt.plot(data[-320:-1])

df.tail(320)

"""
df.set_index('date',inplace=True)
data=df['counts'].tail(480)
Value=data.index[0]
from datetime import datetime, timedelta
Pdates=[]
for y in range(1,480):
 Pdates.append(datetime.strptime(Value,'%Y-%m-%d %H:%M:%S')+timedelta(hours=y))
CC=[0 for i in range(len(Pdates))]
adates=pd.DataFrame({'date':Pdates,'c':CC}) 
data=data.to_frame()
data.reset_index(inplace=True)
adates['date']=adates['date'].astype('str')
actual_data=pd.merge(data,adates,how='right',on='date')
actual_data['counts'].fillna(-1,inplace=True)
actual_data.set_index('date',inplace=True)
data=actual_data['counts'].to_frame()

data=data.sort_values(by=['date'])

#data=data.values
#data=data.to_frame()

#data=diff_values

#data=data.values

"""
Rolling Mean Smoothing
"""

from pandas import Series
from matplotlib import pyplot
#series = Series.from_csv('daily-total-female-births.csv', header=0)
# Tail-rolling average transform
rolling = data.rolling(window=2)
rolling_mean = rolling.mean()
print(rolling_mean.head(10))
# plot original and transformed dataset

%matplotlib auto
data.plot(label='original')
rolling_mean.plot(color='red',label='rolling mean')
pyplot.legend()
pyplot.show()



data=rolling_mean

data.dropna(inplace=True)

#data=data[0:-1]


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

def timeseries_to_supervised3(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(lag,0,-1)]
	columns.append(df)
	df = concat(columns, axis=1)
	#df.dropna(inplace=True)
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
	X = X.reshape(X.shape[0],X.shape[1],1)
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
		model.reset_states()
	return model



def fit_lstmf(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0],1,X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
		model.reset_states()
	return model


def fit_lstmff(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0],X.shape[1],1)
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=False)
	return model


def fit_lstm_main(train,batch_size,nb_epoch,neurons,patience):
    X,y=train[:,0:-1],train[:,-1]
    X=X.reshape(X.shape[0],X.shape[1],1)
    model=Sequential()
    model.add(LSTM(neurons,batch_input_shape=(batch_size, X.shape[1], X.shape[2]),stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    M=0
    N=0
    MODEL=0
    for i in range(nb_epoch):
        print('Epoch'+' : '+str(i))
        if i==0:
           h=model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history 
           M=d['loss']
           MODEL=model
           H=h
        else:   
           h=model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history
           if d['loss']<M:
              M=d['loss']
              MODEL=model
              H=h
           else:
              N=N+1
        if N>patience:
            break
        
    return MODEL,h,H    


def fit_lstm_main2(train,batch_size,nb_epoch,neurons,patience,loss):
    X,y=train[:,0:-1],train[:,-1]
    X=X.reshape(X.shape[0],X.shape[1],1)
    model=Sequential()
    model.add(LSTM(neurons,batch_input_shape=(batch_size, X.shape[1], X.shape[2]),dropout=0.2,stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    M=0
    N=0
    MODEL=0
    for i in range(nb_epoch):
        print('Epoch'+' : '+str(i))
        if i==0:
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history 
           M=d[loss]
           MODEL=model
           H=h
        else:   
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history
           if d[loss]<M:
              M=d[loss]
              MODEL=model
              H=h
           else:
              N=N+1
        if N>patience:
            break
        
    return MODEL,h,H    


def fit_lstm_main20(train,batch_size,nb_epoch,neurons,patience,loss):
    X,y=train[:,0:-1],train[:,-1]
    X=X.reshape(X.shape[0],X.shape[1],1)
    model=Sequential()
    model.add(LSTM(neurons,return_sequences=True,batch_input_shape=(batch_size, X.shape[1], X.shape[2]),dropout=0.2,stateful=True))
    model.add(LSTM(2,input_shape=(X.shape[1],neurons)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    M=0
    N=0
    MODEL=0
    for i in range(nb_epoch):
        print('Epoch'+' : '+str(i))
        if i==0:
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history 
           M=d[loss]
           MODEL=model
           H=h
        else:   
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history
           if d[loss]<M:
              M=d[loss]
              MODEL=model
              H=h
           else:
              N=N+1
        if N>patience:
            break
        
    return MODEL,h,H    



def fit_lstm_main200(train,batch_size,nb_epoch,neurons,patience,loss):
    X,y=train[:,0:-1],train[:,-1]
    X=X.reshape(X.shape[0],X.shape[1],1)
    model=Sequential()
    model.add(LSTM(neurons,return_sequences=True,batch_input_shape=(batch_size, X.shape[1], X.shape[2]),dropout=0.2,stateful=True))
    model.add(LSTM(neurons,input_shape=(X.shape[1],neurons)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    M=0
    N=0
    MODEL=0
    for i in range(nb_epoch):
        print('Epoch'+' : '+str(i))
        if i==0:
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history 
           M=d[loss]
           MODEL=model
           H=h
        else:   
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history
           if d[loss]<M:
              M=d[loss]
              MODEL=model
              H=h
           else:
              N=N+1
        if N>patience:
            break
        
    return MODEL,h,H    



def fit_lstm_main3(train,batch_size,nb_epoch,neurons,patience,loss):
    X,y=train[:,0:-1],train[:,-1]
    X=X.reshape(X.shape[0],1,X.shape[1])
    model=Sequential()
    model.add(LSTM(neurons,batch_input_shape=(batch_size, X.shape[1], X.shape[2]),dropout=0.2,stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    M=0
    N=0
    MODEL=0
    for i in range(nb_epoch):
        print('Epoch'+' : '+str(i))
        if i==0:
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history 
           M=d[loss]
           MODEL=model
           H=h
        else:   
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history
           if d[loss]<M:
              M=d[loss]
              MODEL=model
              H=h
           else:
              N=N+1
        if N>patience:
            break
        
    return MODEL,h,H    



#THis appears to be wrong!

def fit_lstm100(train, batch_size, nb_epoch, neurons):
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0],X.shape[1],1)
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
      model.fit(X, y, epochs=1, batch_size=batch_size,validation_split=0.2, verbose=1, shuffle=False,callbacks=[es,mc])
      model.reset_states()
    return model      
    
    
#Stateful LSTM make sense?
# fit an LSTM network to training data
def fit_lstm0(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0],X.shape[1],1)
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=False)    
	return modelq


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
#DATA=DATA.to_frame()


#DATA=DATA.to_frame()


DATA.reset_index(inplace=True)
#DATA['day']=DATA['day'].apply(lambda x :str(x)[0:10])
dates=DATA.iloc[-24:,:1].values
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
train, test = supervised_values[0:-24], supervised_values[-24:]
# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)
"""
np.argwhere(np.isnan(test_scaled))

na_replace = np.vectorize(lambda x: -2 if np.isnan(x) else x)
train_scaled=na_replace(train_scaled)
test_scaled=na_replace(test_scaled)
"""

# fit the model
#lstm_model = fit_lstm100(train_scaled, 1, 100, 100)
#lstm_model = fit_lstm(train_scaled, 1, 100, 100)
#lstm_model=fit_lstm_main(train_scaled, 1, 100, 100)
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
lstm_model = load_model('best_model.h5')
lstm_model=load_model('/Users/lukishyadav/Desktop/Segmentation/supply_demand/saved_model/best_model.h5')
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
#  LINEAR REGRESSION

from sklearn.linear_model import LinearRegression

def algo(model,train):
 X,y=train[:,0:-1],train[:,-1]
 h=model.fit(X,y)
 return h 
n=algo(LinearRegression(),train_scaled)
n.score(test_scaled[:, 0:-1], test_scaled[:, -1])


from sklearn.metrics import mean_squared_error
from math import sqrt

Yp=n.predict(test_scaled[:, 0:-1])
Yp=Yp.reshape(-1,1)
TTT=np.append(test_scaled[:, 0:-1],Yp,axis=1)
TTTT=scaler.inverse_transform(TTT)

YP=TTTT[:, -1]
rms = sqrt(mean_squared_error(YP, test[:, -1]))
print(rms)


plt.plot(YP)
plt.plot(test[:,-1])



lstm_model,h,H=fit_lstm_main200(train_scaled, 1, 500, 100,60,'val_loss')


H.history

#  lstm_model=h

X, y = test_scaled[:, 0:-1], test_scaled[:, -1]
X=X.reshape(X.shape[0],1,X.shape[1])
X=X.reshape(X.shape[0],X.shape[2],1)
yhat =lstm_model.predict(X,batch_size=1)


TTT=np.append(test_scaled[:, 0:-1],yhat,axis=1)
TTTT=scaler.inverse_transform(TTT)

Yhat=TTTT[:, -1]


#Yt=scaler.inverse_transform(y)

epoch_value=000
#raw_values=random_replace(raw_values)
# report performance
rmse = sqrt(mean_squared_error(raw_values[-24:], Yhat))
print('Test RMSE: {0}, Early Stopp Epoch={1}'.format(rmse,epoch_value))
# line plot of observed vs predicted
%matplotlib auto
pyplot.title(label='Test RMSE: %.3f' % rmse)
pyplot.plot(raw_values[-24:],label='Test Data')
pyplot.plot(Yhat,label='Predicted Data')
pyplot.legend()
pyplot.show()


x=test_scaled[0:, 0:-1]
x=train_scaled[0:-1, 0:-1]
x=x[-1]

"""

Recursive multistep forecasting 


"""

def predict_unseen(lstm_model,X,x,N):
    output=[]
    for n in range(N):
        x=x.reshape(1,x.shape[0],1)
        y=lstm_model.predict(x,batch_size=1)
        x=x.reshape(x.shape[1],1)
        Next=np.append(x,y,axis=0)
        NextTT=Next.reshape(1,lag+1)
        Oo=scaler.inverse_transform(NextTT)
        x=Next[-lag:]
        output.append(Oo[:,-1])
    return output


output=predict_unseen(lstm_model,X,x,24)



P=raw_values[-24:-1].reshape(-1,1)


Q=raw_values[0:-24].reshape(-1,1)


LL=np.array(output)
PP=np.append(P,output,axis=0)

PPP=np.append(Q,output,axis=0)

plt.plot(output)

%matplotlib auto
import matplotlib.pyplot as plt
plt.plot(PP,label='predicted')
plt.plot(raw_values,label='original')
plt.legend()



%matplotlib auto
import matplotlib.pyplot as plt
plt.plot(PPP,label='predicted')
plt.plot(raw_values,label='original')
plt.legend()

%matplotlib auto
import matplotlib.pyplot as plt
plt.plot(output,label='predicted')
plt.plot(raw_values[-24:],label='original')
plt.legend()

plt.plot(output)
plt.plot(raw_values[-24:])



from datetime import datetime, timedelta
timedelta(hours=9)


pdates=[]
for y in range(1,len(dates)+1):
 pdates.append(datetime.strptime(dates[-1][0][0:19],'%Y-%m-%d %H:%M:%S')+timedelta(hours=y))
 

output[0][0]  

for i,u in enumerate(output):
    output[i]=output[i][0]
    
    
FDD=pd.DataFrame({'date':pdates,'p_demand':output})

FDD.to_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/data/19aug_8_24hrs_predictions.csv',index=False)
 
FDD['date']=FDD['date'].astype(str)

FDD=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/data/19aug_8_24hrs_predictions.csv')

pf=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/data/Darwin_Demand_19_8_done.csv')


OO=pd.merge(pf,FDD,on='date',how='right')

OO.fillna(0,inplace=True)

OOO=OO.sort_values(by=['date'])
OOO.reset_index(inplace=True)
OOO.drop(['index'],axis=1,inplace=True)
OOO.reset_index(inplace=True)

%matplotlib auto
import matplotlib.pyplot as plt
plt.scatter(OOO.index,OOO.counts,label='actual')
plt.scatter(OOO.index,OOO.p_demand,label='predicted')
plt.legend()






"""

Correct Inverse Differencing

"""


X=[1,2,3,4,5,6]
dv = difference(X, 1)

Y = pd.Series(1, index=range(0,6))
cumsum = dv.cumsum()


y=Y.add(cumsum)

print predictions_ARIMA_diff_cumsum.head()


1,n

n,1


5*n -7*n




timeseps features

""""

1   2
2   3
3   4
4   5
5    6
6    7


1    2 3 4 5 6 7 









#import my_module
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

from datetime import datetime


#df=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/supply_demand/supply_demand_counts_20190501_20190606.csv')
df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/data_big/data/quadrant_0/timescale_180/hex_edge_174.376m_quantile_3_daily.csv')




LL=len(df.columns)

LLL=[str(i) for i in range((LL-1))]

df.columns=['date']+LLL

key=len(LLL)-2
#key=0
metric='daily'


def convert(x):
    if metric=='hours':
        import re 
      
        # Function to extract all the numbers from the given string 
        def getNumbers(str): 
            array = re.findall(r'[0-9]+', str) 
            return array
        E=getNumbers(x)
        #E=eval(x[14:27])
        #B=(''.join([n for n in x[29:32] if n.isdigit()]))
        #tup=[E[0],E[1],E[2],B]
        tup=[int(E[0]),int(E[1]),int(E[2]),int(E[3])]
        tup=tuple(tup)
        return datetime(tup[0],tup[1],tup[2],tup[3])
    else:
        fmt = '%Y-%m-%d'
        return datetime.strptime(x[0:10], fmt)

df['date']=df['date'].apply(convert)

"""
%matplotlib auto
from matplotlib import pyplot as plt
plt.plot(df.iloc[:,1])
plt.scatter(list(range(len(df))),df.iloc[:,1])
"""

"""
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

"""
import matplotlib.pyplot as plt
plt.plot(data[-320:-1])

df.tail(320)

"""

#import inspect
#inspect.getfullargspec(timedelta) 






df['counts']=df[str(key)]

if metric=='hours':
 rang=240
 N=24
 
else:
    rang=90
    N=4

df.set_index('date',inplace=True)
#data=df['counts'].tail(rang)

data=df['counts']


Value=data.index[0]
#Value=data.index[18]
#eval(Value)
from datetime import datetime, timedelta
Pdates=[]
for y in range(0,len(data)):
 if metric=='hours':  
   Pdates.append(Value+timedelta(hours=y))
 else:  
   Pdates.append(Value+timedelta(days=y))  
   
"""
from datetime import datetime
datetime.strftime(Value)
datetime.parse(Value)
'{0.month}/{0.day}/{0.year} {0.}'.format(Value)
x=Value
"""


CC=[0 for i in range(len(Pdates))]
adates=pd.DataFrame({'date':Pdates,'c':CC}) 
data=data.to_frame()
data.reset_index(inplace=True)
adates['date']=adates['date'].astype('str')
data['date']=data['date'].astype(str)
actual_data=pd.merge(data,adates,how='right',on='date')
import statistics
val=statistics.mode(actual_data['counts'])
actual_data['counts'].fillna(val,inplace=True)
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


s_window=2

original=data[s_window-1:].head(rang)



from pandas import Series
from matplotlib import pyplot as plt
#series = Series.from_csv('daily-total-female-births.csv', header=0)
# Tail-rolling average transform
rolling = data.rolling(window=s_window)
rolling_mean = rolling.mean()
print(rolling_mean.head(10))
# plot original and transformed dataset

"""
%matplotlib auto
plt.plot(data,label='original')
plt.plot(rolling_mean,color='red',label='rolling mean')
plt.legend()
plt.show()

plt.scatter(list(range(len(data))),data,label='original')
plt.scatter(list(range(len(data))),rolling_mean,color='red',label='rolling mean')
plt.legend()
plt.show()
"""

data=rolling_mean

data.dropna(inplace=True)


DATA=data.head(rang)




from pyramid.arima import auto_arima


model = auto_arima(DATA[0:-N], trace=True, error_action='ignore', suppress_warnings=True)
model.fit(DATA[0:-N])


d=model.get_params()

P=d['order'][0]




if P==0:
 forecast=[]
 train=original[:-N]
 
 buffer=list(train.tail(3).values.reshape(1,3)[0])
 for x in range(N):
    Buffer=pd.DataFrame(buffer)
    rolling = Buffer.rolling(window=3)
    rolling_mean = rolling.mean() 
    forecast.append(rolling_mean.iloc[-1].values[0])
    buffer.append(rolling_mean.iloc[-1].values[0])


 forecast=list(np.around(np.array(forecast),0))

 %matplotlib auto
 import matplotlib.pyplot as plt
 plt.plot(forecast,label='predicted')
 plt.plot(original[-N:].values,label='original')
 plt.legend()
 plt.title('Rolling Mean Forecast.   P value:'+str(P))
 plt.xlabel('Progression in '+str(metric))
 plt.ylabel('Demand')
#plt.xticks(range(len()))


"""
forecast=list(np.around(np.array(forecast),0))
act=np.around(np.array(DATA[-N:]),0)
act=original[1:]
%matplotlib auto
import matplotlib.pyplot as plt
plt.plot(DATA[-N:],label='predicted')
plt.plot(act[-N:],label='original')
plt.legend()
"""


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


def fit_lstm_main20(train,n_lag, n_seq, n_batch,nb_epoch,n_neurons,patience,loss):
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0],X.shape[1],1)
    model=Sequential()
    model.add(LSTM(n_neurons,return_sequences=True,batch_input_shape=(n_batch, X.shape[1], X.shape[2]),dropout=0.2,stateful=True))
    model.add(LSTM(n_neurons,input_shape=(X.shape[1],n_neurons)))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    M=0
    N=0
    MODEL=0
    for i in range(nb_epoch):
        print('Epoch'+' : '+str(i))
        if i==0:
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=n_batch, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history 
           M=d[loss]
           MODEL=model
           H=h
        else:   
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=n_batch, verbose=1, shuffle=False)
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

def fit_lstm_main200(train,test,n_lag, n_seq, n_batch,nb_epoch,n_neurons,patience,loss):
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0],X.shape[1],1)
    test_x,test_y=test[:,0:n_lag],test[:,n_lag:]
    test_x=test_x.reshape(test_x.shape[0],test_x.shape[1],1)
    model=Sequential()
    model.add(LSTM(n_neurons,return_sequences=True,batch_input_shape=(n_batch, X.shape[1], X.shape[2]),dropout=0.2,stateful=True))
    model.add(LSTM(n_neurons,input_shape=(X.shape[1],n_neurons)))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    M=0
    N=0
    MODEL=0
    for i in range(nb_epoch):
        print('Epoch'+' : '+str(i))
        if i==0:
           h=model.fit(X, y, epochs=1,validation_data=(test_x,test_y),batch_size=n_batch, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history 
           M=d[loss]
           MODEL=model
           H=h
        else:   
           h=model.fit(X, y, epochs=1,validation_data=(test_x,test_y),batch_size=n_batch, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history
           if d[loss]<M:
              M=d[loss]
              MODEL=model
              H=h
              N=0
           else:
              N=N+1
        if N>patience:
            break
        
    return MODEL,h,H   



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
	X=series[-1, 0:n_lag]
    #X=series[:, 0:n_lag]
    
	X=X.reshape(1,X.shape[0],1)
	forecast = model.predict(X, batch_size=n_batch)
	#forecasts.append(forecast)
	return forecast

# evaluate the persistence model
def make_train_forecasts(model, n_batch, series, n_lag, n_seq):
	forecasts = list()
	#for i in range(len(test)):
	#X, y = test[-1, 0:n_lag], test[-1, n_lag:]
	X=series[-1, 0:n_lag]
    #X=series[:, 0:n_lag]
    
	X=X.reshape(1,X.shape[0],1)
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


#series=FD['counts'].to_frame()

series=DATA

n_lag = P
n_seq = N
n_test = N
n_epochs = 50
n_batch = 1
n_neurons = 100
patience=50
loss='val_loss'
# prepare data
scaler,train, test = prepare_data(series, n_test, n_lag, n_seq)
# fit model
#model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)

#model = fit_lstm2(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)


#model,h,H=fit_lstm_main20(train, n_lag, n_seq, n_batch, n_epochs, n_neurons,patience,loss)


model,h,H=fit_lstm_main200(train,test, n_lag, n_seq, n_batch, n_epochs, n_neurons,patience,loss)


H.history
print(model.summary())


# make forecasts
forecasts = make_forecasts2(model, n_batch, train, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
%matplotlib auto
plot_forecasts(series, forecasts, n_test+2)

from matplotlib import pyplot as plt
plt.plot(forecasts[0],label='predicted')
plt.plot(series[-n_test:],label='original') 

plt.legend()
#plt.plot(n_test+2)

#unseen data

"""
predictions=make_unseen_forecasts(model, n_batch, test, n_lag, n_seq)
outputs=scaler.inverse_transform(predictions)
train_data
plt.plot(outputs[0])
"""

#Train_test Verification 

predictions=make_unseen_forecasts(model, n_batch, train, n_lag, n_seq)
outputs=scaler.inverse_transform(predictions)

OOO=list(np.around(np.array(outputs[0]),0))

plt.plot(OOO,label='predicted')
plt.plot(original[-n_test:].values,label='original')   
plt.legend()    
plt.title('LSTM forecast direct multistep   P value:'+str(P))
plt.xlabel('Progression in '+str(metric))
plt.ylabel('Demand')
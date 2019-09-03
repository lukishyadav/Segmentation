#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:44:50 2019

@author: lukishyadav
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


df=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/supply_demand/supply_demand_counts_20190501_20190606.csv')

df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/Darwin_Demand.csv')

df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/data/Darwin_Demand_19_8.csv')



DF=df.copy()


DF['day']=DF['date'].apply(lambda x:datetime.datetime.strptime(x[0:10],'%Y-%m-%d'))


FD=DF.groupby(['day']).sum(name='Counts')


#import pandas
#import matplotlib.pyplot as plt

#plt.plot(DF['Supply_Count'])

['Unnamed: 0', 'Date', 'Appopen_Count', 'Supply_Count', 'Rental_Count',
       'hour']


data=DF['counts']

data=FD['counts']

data=df['counts'].tail(320)
data=data[0:-1]




"""
For hourly data
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
actual_data['counts'].fillna(1,inplace=True)


actual_data.set_index('date',inplace=True)
data=actual_data['counts'].to_frame()

data=data.sort_values(by=['date'])


data=data[0:-1]




#divide into train and validation set
train = data[:int(0.67*(len(data)))]
valid = data[int(0.67*(len(data))):]

#plotting the data
train.plot(label='Training Data',legend=True)
valid.plot(label='Testing Data',legend=True)

data.plot()

import matplotlib.pyplot as plt
plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.legend()
plt.show()



from pyramid.arima import auto_arima


model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)



from pandas import Series
from matplotlib import pyplot
#series = Series.from_csv('daily-total-female-births.csv', header=0)
# Tail-rolling average transform
rolling = data.rolling(window=2)
rolling_mean = rolling.mean()
print(rolling_mean.head(10))


%matplotlib auto
pyplot.plot(data,label='original')
pyplot.plot(rolling_mean,color='red',label='rolling mean')
pyplot.legend()
pyplot.show()




model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(rolling_mean)


forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(valid, forecast))
print('Test RMSE: %.3f' % rmse)


"""
Decomposing Trend, Seasonality and Residuals. 
"""


from random import randrange
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
# =============================================================================
# series = [i+randrange(10) for i in range(1,100)]
# =============================================================================
result = seasonal_decompose(data, model='multiplicative', freq=1)
result.plot()
pyplot.show()




from statsmodels.tsa.statespace.sarimax import SARIMAX



forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(valid, forecast))
print('Test RMSE: %.3f' % rmse)


import matplotlib.pyplot as plt
%matplotlib auto
#%matplotib inline

#plot the predictions for validation set
plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.legend()
plt.show()
#plt.ion()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:31:19 2019

@author: lukishyadav
"""

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/Users/lukishyadav/Desktop/segmentation/supply_demand_main/codes/flow')

import sd_module as sd

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
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


#/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/data_big/data/quadrant_0/timescale_30/

## supply_demand_main/data_big/data/quadrant0/timescale_30/hex_edge_0.51m_quantile_0_daily.csv


#df=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/supply_demand/supply_demand_counts_20190501_20190606.csv')
#df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/Darwin_Demand.csv')
#df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/data_big/data/quadrant_0/timescale_90/hex_edge_24.911m_quantile_3_daily.csv')
#174.376,  461.355
df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/data_big/data/quadrant_0/timescale_180/hex_edge_174.376m_quantile_1_hourly.csv')




LL=len(df.columns)

LLL=[str(i) for i in range((LL-1))]

df.columns=['date']+LLL

key=len(LLL)-2
#key=0
metric='hours'


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


#df.reset_index(inplace=True)
df['day']=df['date'].apply(lambda x:x.day)
df['month']=df['date'].apply(lambda x:x.month)
df['year']=df['date'].apply(lambda x:x.year)
df['hour']=df['date'].apply(lambda x:x.hour)
df['weekday']=df['date'].apply(lambda x:x.weekday())


def hbin(x):
    if x<=5:
        return 1
    elif x<=11:
        return 2
    elif x<=17:
        return 3
    else:
        return 4

df['hchunk']=df['hour'].apply(hbin)

#LLL.append('weekday')


dic={LLL[i]:'sum' for i in range(len(LLL))}
dic.update({'weekday':'min'})


DDFF=df.groupby(['year','month','day','hchunk']).agg(dic)


#DDFF=df.groupby(['day','month','year','hchunk']).agg({'weekday':'count'})

#DDFF.reset_index(inplace=True)



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






DDFF['counts']=DDFF[str(key)]

if metric=='hours':
 rang=240
 N=24
 
else:
    rang=90
    N=7

#df.set_index('date',inplace=True)
#data=df['counts'].tail(rang)

data=DDFF['counts']


"""


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
import collections
try:
 val=statistics.mode(actual_data['counts'])
except statistics.StatisticsError:
 val=collections.Counter(actual_data['counts'])  

if val==dict: 
 val=max(val, key=val.get)
 
actual_data['counts'].fillna(val,inplace=True)
actual_data.set_index('date',inplace=True)
data=actual_data['counts'].to_frame()

data=data.sort_values(by=['date'])


"""

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
data=data.to_frame()

DATA=data.head(rang)




from pyramid.arima import auto_arima


model = auto_arima(DATA[0:-N], trace=True, error_action='ignore', suppress_warnings=True)
#model.fit(DATA[0:-N])

foreca = model.predict(n_periods=N)
foreca=list(foreca)


d=model.get_params()

P=d['order'][0]



"""
plt.plot(foreca)
plt.plot(original[-N:].values)


import matplotlib

m=28

labls=DDFF['weekday'].head(rang).values

colors = {0:'b',1:'g',2:'r',3:'c',4:'m',5:'y',6:'k'}

plt.scatter(list(range(len(original[-m:]))),original[-m:],c=labls[-m:], cmap=matplotlib.colors.ListedColormap(colors)
)

scatter_x=np.array(list(range(len(original[-m:]))))
scatter_y=np.array(list(original[-m:].values))
cdict={0:'b',1:'g',2:'r',3:'c',4:'m',5:'y',6:'k'}
group=DDFF['weekday'].head(rang).values[-m:]


fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 100)
ax.legend()
plt.show()




plt.scatter(list(range(len(foreca))),foreca,label="Forecasted")
plt.scatter(list(range(len(foreca))),original[-N:].values,label='Original')
plt.legend()

"""




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
 
 vector1 = np.array(forecast).reshape(1,-1)
 vector2 = np.array(foreca)

 sum_vector = (vector1 + vector2)/2

 #output=list(np.around(np.array(output),0).reshape(1,-1)[0])

 ensemble=list(np.around(np.array(sum_vector),0).reshape(1,-1)[0])
 aa_forecast=np.around(vector2,0)
 
 %matplotlib auto
 import matplotlib.pyplot as plt
 plt.plot(forecast,label='predicted')
 plt.plot(original[-N:].values,label='original')
 plt.plot(ensemble,label='ensemble')
 plt.plot(aa_forecast,label='AutoArima Forecast')
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



# load dataset
#series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

lag=P
#lag=3
series=DATA
DATA=series.copy()
#DATA=DATA.to_frame()


#DATA=DATA.to_frame()




import pickle

def see_unseen(lag,series,P_value,metric,foreca):
    DATA.reset_index(inplace=True)
    #DATA['day']=DATA['day'].apply(lambda x :str(x)[0:10])
    dates=DATA.iloc[-N:,:1].values
    # transform data to be stationary
    raw_values = series.values
    diff_values = sd.difference(raw_values, 1)
    #import matplotlib.pyplot as plt
    #plt.plot(diff_values)
    # transform data to be supervised learning
    #supervised = timeseries_to_supervised2(diff_values, lag)
    supervised = sd.timeseries_to_supervised2(series, lag)
    supervised_values = supervised.values
    # split data into train and test-sets
    train, test = supervised_values[0:-N], supervised_values[-N:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = sd.scale(train, test,(-1, 1))
    
    #lstm_model,h,H=fit_lstm_main200(train_scaled, 1, 500, 100,100,'val_loss')
    
    #lstm_model,h,H=fit_lstm_main2000(train_scaled, 1, 500, 100,100,'loss')
    lstm_model,h,H=sd.fit_lstm_main2000(train_scaled,test_scaled, 1, 500, 100,100,'val_loss')

    #filename = '/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/finalized_model.sav'
    #pickle.dump(lstm_model, open(filename, 'wb'))
    #H.history
    
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
    rmse = sqrt(mean_squared_error(raw_values[-N:], Yhat))
    print('Test RMSE: {0}, Early Stopp Epoch={1}'.format(rmse,epoch_value))
    # line plot of observed vs predicted
    
    
    #x=test_scaled[0:, 0:-1]
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
    
    
    
    
    output=predict_unseen(lstm_model,X,x,N)
    
    
    
    P=raw_values[-N:-1].reshape(-1,1)
    
    
    Q=raw_values[0:-N].reshape(-1,1)
    
    
    LL=np.array(output)
    PP=np.append(P,output,axis=0)
    
    PPP=np.append(Q,output,axis=0)
    
    #plt.plot(output)
    
    %matplotlib auto
    import matplotlib.pyplot as plt
    #plt.plot(output,label='predicted')
    #plt.plot(raw_values[-N:],label='original')
    #plt.legend()
    
    
    vector1 = np.array(output).reshape(1,-1)
    vector2 = np.array(foreca)

    sum_vector = (vector1 + vector2)/2

    #output=list(np.around(np.array(output),0).reshape(1,-1)[0])

    ensemble=list(np.around(np.array(sum_vector),0).reshape(1,-1)[0])
    
    aa_forecast=np.around(vector2,0)
   
    
    plt.plot(output,label='predicted')
    plt.plot(original[-N:].values,label='original')
    plt.plot(ensemble,label='ensemble')
    plt.plot(aa_forecast,label='AutoArima Forecast')
    plt.legend()
    plt.title('LSTM forecast recursive multistep   P value:'+str(P_value))
    #plt.xlabel('Progression in '+str(metric))
    plt.xlabel('Progression in '+'quarter day')
    plt.ylabel('Demand')
    return scaler,lstm_model,test_scaled
    

if P!=0:
    scaler,lstm_model,test_scaled=see_unseen(lag,series,P,metric,foreca)
    filename = 'finalized_model.sav'
    d={'model':lstm_model,'scaler':scaler,'test_scaled':test_scaled}
    pickle.dump(d, open(filename, 'wb'))
     
    # some time later...
     
    # load the model from disk
    


import pickle
filename = 'finalized_model.sav'
d = pickle.load(open(filename, 'rb'))
scaler=d['scaler']
test_scaled=d['test_scaled']
lstm_model=d['model']


DATA=data.head(rang+24)


x=test_scaled[0:, 0:-1]
x=x[-1]
    
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
    
    
X=1

output=predict_unseen(lstm_model,X,x,N)

"""

P=raw_values[-N:-1].reshape(-1,1)


Q=raw_values[0:-N].reshape(-1,1)


LL=np.array(output)
PP=np.append(P,output,axis=0)

PPP=np.append(Q,output,axis=0)
"""
model = auto_arima(DATA[0:-N], trace=True, error_action='ignore', suppress_warnings=True)
#model.fit(DATA[0:-N])

foreca = model.predict(n_periods=N)
foreca=list(foreca)


vector1 = np.array(output).reshape(1,-1)
vector2 = np.array(foreca)

sum_vector = (vector1 + vector2)/2

#output=list(np.around(np.array(output),0).reshape(1,-1)[0])

ensemble=list(np.around(np.array(sum_vector),0).reshape(1,-1)[0])

aa_forecast=np.around(vector2,0)
   

plt.plot(output,label='predicted')
plt.plot(original[-N:].values,label='original')
plt.plot(ensemble,label='ensemble')
plt.plot(aa_forecast,label='AutoArima Forecast')
plt.plot(foreca,label='Float AutoArima Forecast')
plt.legend()
plt.title('LSTM forecast recursive multistep   P value:'+str(P))
#plt.xlabel('Progression in '+str(metric))
plt.xlabel('Progression in '+'quarter day')
plt.ylabel('Demand')
#return scaler,lstm_model


%matplotlib auto
plt.scatter(list(range(len(output))),output,label='predicted')
plt.scatter(list(range(len(output))),original[-N:].values,label='original')
plt.scatter(list(range(len(output))),ensemble,label='ensemble')
plt.scatter(list(range(len(output))),aa_forecast,label='AutoArima Forecast')
plt.scatter(list(range(len(output))),foreca,label='Float AutoArima Forecast')
plt.legend()
plt.title('LSTM forecast recursive multistep   P value:'+str(P))
#plt.xlabel('Progression in '+str(metric))
plt.xlabel('Progression in '+'quarter day')
plt.ylabel('Demand')





"""

NEW DATA CHECK



"""



DATA2=data.tail(rang)
series2=DATA2

raw_values = series2.values

supervised = timeseries_to_supervised2(series, lag)
supervised_values = supervised.values
# split data into train and test-sets
train, test = supervised_values[0:-N], supervised_values[-N:]
# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test,(-1, 1))



#x=test_scaled[0:, 0:-1]
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

output=predict_unseen(lstm_model,X,x,N)

P=raw_values[-N:-1].reshape(-1,1)

Q=raw_values[0:-N].reshape(-1,1)

LL=np.array(output)
PP=np.append(P,output,axis=0)

PPP=np.append(Q,output,axis=0)

#plt.plot(output)

%matplotlib auto
import matplotlib.pyplot as plt
#plt.plot(output,label='predicted')
#plt.plot(raw_values[-N:],label='original')
#plt.legend()


vector1 = np.array(output).reshape(1,-1)
vector2 = np.array(foreca)

sum_vector = (vector1 + vector2)/2

#output=list(np.around(np.array(output),0).reshape(1,-1)[0])

ensemble=list(np.around(np.array(sum_vector),0).reshape(1,-1)[0])

aa_forecast=np.around(vector2,0)
   

plt.plot(output,label='predicted')
plt.plot(original[-N:].values,label='original')
plt.plot(ensemble,label='ensemble')
plt.plot(aa_forecast,label='AutoArima Forecast')
plt.legend()
plt.title('LSTM forecast recursive multistep   P value:'+str(P_value))
#plt.xlabel('Progression in '+str(metric))
plt.xlabel('Progression in '+'quarter day')
plt.ylabel('Demand')




"""
#Unseen Verification
 
 

"""
sum_vector/2
output[1]

foreca=list(foreca)

vector1 = np.array(output)
vector2 = np.array(foreca)

sum_vector = vector1 + vector2

list(sum_vector)


np.argwhere(np.isnan(test_scaled))

na_replace = np.vectorize(lambda x: -2 if np.isnan(x) else x)
train_scaled=na_replace(train_scaled)
test_scaled=na_replace(test_scaled)



#  LINEAR REGRESSION

from sklearn.linear_model import LinearRegression

def algo(model,train):
 X,y=train[:,0:-1],train[:,-1]
 h=model.fit(X,y)
 return h 
mod=algo(LinearRegression(),train_scaled)
mod.score(test_scaled[:, 0:-1], test_scaled[:, -1])


from sklearn.metrics import mean_squared_error
from math import sqrt

Yp=mod.predict(test_scaled[:, 0:-1])
Yp=Yp.reshape(-1,1)
TTT=np.append(test_scaled[:, 0:-1],Yp,axis=1)
TTTT=scaler.inverse_transform(TTT)

YP=TTTT[:, -1]
rms = sqrt(mean_squared_error(YP, test[:, -1]))
print(rms)


plt.plot(YP,label='Prediction')
plt.plot(test[:,-1],label='Original')
plt.legend()


model=n
x=test_scaled[0:, 0:-1]
x=train_scaled[0:-1, 0:-1]
x=x[-1]
N=24

def predict_unseen2(model,X,x,N):
    output=[]
    for n in range(N):
        x=x.reshape(1,-1)
        y=model.predict(x)
        x=x.reshape(x.shape[1],1)
        y=y.reshape(-1,1)
        Next=np.append(x,y,axis=0)
        NextTT=Next.reshape(1,lag+1)
        Oo=scaler.inverse_transform(NextTT)
        x=Next[-lag:]
        output.append(Oo[:,-1])
    return output

output=predict_unseen2(mod,X,x,24)


%matplotlib auto
import matplotlib.pyplot as plt
plt.plot(output,label='predicted')
plt.plot(raw_values[-24:],label='original')
plt.legend()






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

#plt.plot(output)
#plt.plot(raw_values[-24:])



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

plt.plot(series.iloc[-N:])


plt.plot(df[str(key)].values)

plt.plot(df[s_window-1:].head(rang))

"""








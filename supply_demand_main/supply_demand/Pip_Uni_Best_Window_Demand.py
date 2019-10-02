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
import os
from matplotlib import pyplot as plt


#df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/data_big/data/quadrant_0/timescale_90/hex_edge_24.911m_quantile_3_daily.csv')
#174.376,  461.355, 1220.63


file='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/data_big/data/quadrant_0/timescale_30/hex_edge_461.355m_quantile_3_hourly.csv'

file='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/data_big/data/quadrant_0/timescale_30/hex_edge_461.355m_quantile_4_hourly.csv'


df=pd.read_csv(file)

#Supply_Demand_Data_Sync
dfs=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply/darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_quantile_4_hourly.csv')

df=pd.merge(df,dfs['timeseries'],on='timeseries',how='inner')


import re
result = re.search('ge_(.*)_hourly', file)
print(result.group(1))

qua=result.group(1)

store=df.columns
mname=file[-35:-4]


LL=len(df.columns)

LLL=[str(i) for i in range((LL-1))]

df.columns=['date']+LLL

key=len(LLL)-2
key=0

dpath='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/images/demand/'+result.group(1)+'_key_'+str(key)+'_'+store[key+1][1:-1]
os.mkdir(dpath)


metric='hours'
agg=6

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
Calculations to replace missing hour stamps by mode value

"""


Value=df['date'].iloc[0]
#Value=data.index[18]
#eval(Value)
from datetime import datetime, timedelta
Pdates=[]
for y in range(0,len(df)):
 if metric=='hours':  
   Pdates.append(Value+timedelta(hours=y))
 else:  
   Pdates.append(Value+timedelta(days=y))  


CC=[0 for i in range(len(Pdates))]
adates=pd.DataFrame({'date':Pdates,'c':CC}) 
data=df
#data.reset_index(inplace=True)
#adates['date']=adates['date'].astype('str')
#data['date']=data['date'].astype(str)
actual_data=pd.merge(data,adates,how='right',on='date')
import statistics
import collections
try:
 val=statistics.mode(actual_data[str(key)])
except statistics.StatisticsError:
 val=collections.Counter(actual_data[str(key)])  

if val==dict: 
 val=max(val, key=val.get)
 
actual_data[str(key)].fillna(val,inplace=True)
#actual_data.set_index('date',inplace=True)
#data=actual_data['counts'].to_frame()
data=data.sort_values(by=['date'])

df=actual_data.copy()



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
#import inspect
#inspect.getfullargspec(timedelta) 

DDFF['counts']=DDFF[str(key)]

if metric=='hours':
 rang=int(240/agg)
 N=int(24/agg)
 
else:
    rang=90
    N=7

#df.set_index('date',inplace=True)
#data=df['counts'].tail(rang)

data=DDFF['counts']


"""
Rolling Mean Smoothing
"""
windows=[1,2,3,4]

Mdict={}

Cdict={}

Pdict={}

Fdict={}

for win in windows:
    
    data=DDFF['counts']
    #s_window=4
    
    s_window=win
    
    #original=data[s_window-1:].head(rang)
    #DATA=data.head(rang+N)
    
    original=data[4-1:].head(rang+N).to_frame()
    
    original.reset_index(inplace=True)
    
    original=original[['counts']]
    
    
    if win!=0:
        from pandas import Series
        from matplotlib import pyplot as plt
        #series = Series.from_csv('daily-total-female-births.csv', header=0)
        # Tail-rolling average transform
        rolling = data.rolling(window=s_window)
        rolling_mean = rolling.mean()
        print(rolling_mean.head(10))
        # plot original and transformed dataset
        
    else:
        #rolling_mean=data[4-1:].head(rang)
        rolling_mean=data
    
    """
    %matplotlib auto
    plt.plot(data,label='original')
    plt.plot(rolling_mean.values,color='red',label='rolling mean')
    plt.legend()
    plt.show()
    
    plt.scatter(list(range(len(data))),data,label='original')
    plt.scatter(list(range(len(data))),rolling_mean,color='red',label='rolling mean')
    plt.legend()
    plt.show()
    """
    
    data2=data
    
    data=rolling_mean
    
    """To check how is rolling mean doing
    plt.plot(data2.head(rang).values,label='original')
    plt.plot(data.head(rang).values,color='red',label='rolling mean')
    plt.legend()
    plt.show()
    
    
    """
    data=data.iloc[3:]
    data.dropna(inplace=True)
    data=data.to_frame()
    
    DATA=data.head(rang)
    
    DATA.reset_index(inplace=True)
    
    DATA=DATA[['counts']]
    
    from pyramid.arima import auto_arima
    
    model = auto_arima(DATA[0:-N], trace=True, error_action='ignore', suppress_warnings=True)
    #model.fit(DATA[0:-N])
    
    foreca = model.predict(n_periods=N)
    foreca=list(foreca)
    
    d=model.get_params()
    
    P=d['order'][0]
    
    
    if P==0:
     forecast=[]
     train=original[:-N]
     
     buffer=list(train.tail(win).values.reshape(1,win)[0])
     for x in range(N):
        Buffer=pd.DataFrame(buffer)
        rolling = Buffer.rolling(window=win)
        rolling_mean = rolling.mean() 
        forecast.append(rolling_mean.iloc[-1].values[0])
        buffer.append(rolling_mean.iloc[-1].values[0])
    
    
     forecast=list(np.around(np.array(forecast),0))
     
     vector1 = np.array(forecast).reshape(1,-1)
     
     DATA=data.head(rang+N)
     
     model = auto_arima(DATA[0:-N], trace=True, error_action='ignore', suppress_warnings=True)
     #model.fit(DATA[0:-N])
    
     foreca = model.predict(n_periods=N)
     
     vector2 = np.array(foreca)
    
     sum_vector = (vector1 + vector2)/2
    
     #output=list(np.around(np.array(output),0).reshape(1,-1)[0])
    
     ensemble=list(np.around(np.array(sum_vector),0).reshape(1,-1)[0])
     aa_forecast=np.around(vector2,0)
     output=np.around(vector2,0)
     
     rmse={x:sqrt(mean_squared_error(original[-N:].values, eval(x))) for x in ['aa_forecast','ensemble','output']}
    
     # Selecting minimum RNSE value model
     chosen_one=min(rmse, key=rmse.get)   
    
     from statistics import mean
    
     #mean_one=mean(rmse.values())
     mean_one=rmse[chosen_one]
    
     Mdict[win]=mean_one   #Windows best RMSE
    
     Cdict[win]=chosen_one   #Each models best RMSE
    
     Pdict[win]=eval(chosen_one)  # Each windows prediction

     
     #%matplotlib auto
     import matplotlib.pyplot as plt
     plt.plot(output,label='Rolling Mean Forecast')
     plt.plot(original[-N:].values,label='Original')
     plt.plot(ensemble,label='Ensemble')
     plt.plot(aa_forecast,label='AutoArima Forecast')
     plt.legend()
     plt.title('lat_lng:'+store[key+1][1:-1]+' P_Value: '+str(P))
     plt.xlabel('Progression in '+str(metric))
     plt.ylabel('Demand')
     plt.savefig(dpath+'/Output_window'+str(win)+'.png')
     plt.clf()
   
     
    #plt.xticks(range(len()))
    
    
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
        #DATA.reset_index(inplace=True)
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
        
        #%matplotlib auto
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
       
        """
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
        return scaler,lstm_model,test_scaled
        
    
    if P!=0:
        scaler,lstm_model,test_scaled=see_unseen(lag,series,P,metric,foreca)
        filename = '/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/codes/flow/finalized_model'+mname+'.sav'
        d={'model':lstm_model,'scaler':scaler,'test_scaled':test_scaled}
        pickle.dump(d, open(filename, 'wb'))
         
        # some time later...
         
        # load the model from disk
        
    
    
        import pickle
        filename = '/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/codes/flow/finalized_model'+mname+'.sav'
        d = pickle.load(open(filename, 'rb'))
        scaler=d['scaler']
        test_scaled=d['test_scaled']
        lstm_model=d['model']
        
        
        DATA=data.head(rang+N)
        
        DATA.reset_index(inplace=True)
    
        DATA=DATA[['counts']]
    
        
        
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
        
        ooutput=np.around(output,0).reshape(1,-1)[0]
        
        rmse={x:sqrt(mean_squared_error(original[-N:].values, eval(x))) for x in ['aa_forecast','ensemble','ooutput']}
        
        # Selecting minimum RNSE value model
        chosen_one=min(rmse, key=rmse.get)   
        
        from statistics import mean
        
        #mean_one=mean(rmse.values())
        mean_one=rmse[chosen_one]
        
        Mdict[win]=mean_one
        
        Cdict[win]=chosen_one
        
        Pdict[win]=eval(chosen_one)
        
        Fdict[win]=lstm_model
    
        plt.plot(output,label='LSTM predicted')
        plt.plot(original[-N:].values,label='original')
        plt.plot(ensemble,label='Ensemble')
        plt.plot(aa_forecast,label='AutoArima Forecast')
        plt.plot(foreca,label='Float AutoArima Forecast')
        plt.legend()
        plt.title('lat_lng:'+store[key+1][1:-1]+' P_Value: '+str(P))
        #plt.xlabel('Progression in '+str(metric))
        plt.xlabel('Progression in '+'quarter day')
        plt.ylabel('Demand')
        plt.savefig(dpath+'/Output_window'+str(win)+'.png')
        plt.clf()
        #return scaler,lstm_model
        """
        
        #%matplotlib auto
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


mkey=min(Mdict,key=Mdict.get)

['aa_forecast','ensemble','ooutput']
if Cdict[mkey]=='ooutput':
    Cdict[mkey]='LSTM'
elif Cdict[mkey]=='aa_forecast':
    Cdict[mkey]='Auto ARIMA'
elif Cdict[mkey]=='output':
    Cdict[mkey]='Rolling_Mean_Forecast'    
else:
    Cdict[mkey]='Ensemble(LSTM and AutoARIMA)'    

plt.plot(Pdict[mkey],label='chosen model: '+Cdict[mkey]+' Best_Window_Size:'+str(mkey))
plt.plot(original[-N:].values,label='original')
#plt.plot(ensemble,label='ensemble')
#plt.plot(aa_forecast,label='AutoArima Forecast')
#plt.plot(foreca,label='Float AutoArima Forecast')
plt.legend()
plt.title('lat_lng:'+store[key+1][1:-1]+' P_Value: '+str(P))
#plt.xlabel('Progression in '+str(metric))
plt.xlabel('Progression in '+'quarter day')
plt.ylabel('Demand')
plt.savefig(dpath+'/Final_Output_window'+str(mkey)+'.png')
plt.clf()
#return scaler,lstm_model









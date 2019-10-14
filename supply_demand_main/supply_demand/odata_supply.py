#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:31:19 2019

@author: lukishyadav
"""
import time
st=time.time()


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

[174.376,461.355,1220.63]

#file='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/data/supply/darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_quantile_3_hourly.csv'

file='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/data/demand/darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_all_hexes_hourly.csv'


df=pd.read_csv(file)

#Supply_Demand_Data_Sync
dfd=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/data/supply/darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_all_hexes_hourly.csv')

CL=df.columns
CL2=dfd.columns
CL3=set(CL).intersection(CL2)
df=pd.merge(df['timeseries'],dfd[CL3],on='timeseries',how='inner')

cals=list(CL3)
cals.sort(reverse=True)
df=df[cals]

import re
result = re.search('ge_(.*)_hourly', file)
print(result.group(1))

qua=result.group(1)

store=df.columns
mname=file[-35:-4]


LL=len(df.columns)

LLL=[str(i) for i in range((LL-1))]

df.columns=['date']+LLL

#key=len(LLL)-2
#key=0

for key in range(len(LLL)):
#for key in range(37,len(LLL)):    
    
    file='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/data/demand/darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_all_hexes_hourly.csv'


    df=pd.read_csv(file)
    
    #Supply_Demand_Data_Sync
    dfd=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/data/supply/darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_all_hexes_hourly.csv')
    
    CL=df.columns
    CL2=dfd.columns
    CL3=set(CL).intersection(CL2)
    df=pd.merge(df['timeseries'],dfd[CL3],on='timeseries',how='inner')
    
    cals=list(CL3)
    cals.sort(reverse=True)
    df=df[cals]

    import re
    result = re.search('ge_(.*)_hourly', file)
    print(result.group(1))
    
    qua=result.group(1)
    
    store=df.columns
    mname=file[-35:-4]
    
    
    LL=len(df.columns)
    
    LLL=[str(i) for i in range((LL-1))]
    
    df.columns=['date']+LLL



    dpath='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/images/supply/'+result.group(1)+'_key_'+str(key)+'_'+store[key+1][1:-1]
    
    #os.mkdir(dpath)
    
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
     val=dict(collections.Counter(actual_data[str(key)])) 
    
    if type(val)==dict: 
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
    
    allp={}
    
    for win in windows:
        
        data=DDFF['counts']
        #s_window=4
        
        s_window=win
        
        #original=data[s_window-1:].head(rang)
        #DATA=data.head(rang+N)
        
        original=data[4-1:].head(rang+N).to_frame()
        
        original.reset_index(inplace=True)
        
        original=original[['counts']]
        
        od[win]=original.values.reshape(1,-1)[0]
        
        
    op=pd.DataFrame(np.array([str(od)]).T,
                    columns=['original_data'])
    #op.iloc[0]=np.array([str(Mdict)],[str(Cdict)],[str(Pdict)],[str(allp)])
    op.to_csv(dpath+'/original_data_'+str(key)+'.csv')
    #return scaler,lstm_model
     
    
    
print(time.time()-st)  
        
    
    
    
    

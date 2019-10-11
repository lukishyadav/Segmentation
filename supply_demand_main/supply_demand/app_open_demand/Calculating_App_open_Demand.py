#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:01:30 2019

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
from datetime import datetime
from datetime import timedelta
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


#file='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/data_big/data/quadrant_0/timescale_30/hex_edge_461.355m_quantile_3_hourly.csv'

file='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/app_open_demand/appopen_darwin_20190531_20190630.csv'


df=pd.read_csv(file)


df['created_at'] = df['created_at'].apply(lambda x:datetime.strptime(x[0:19], '%Y-%m-%d %H:%M:%S'))
df['flag']=df['customer_id'].apply(lambda x:0 if str(x)=='nan' else 1)

df['minute']=df['created_at'].apply(lambda x:x.minute)
df['hour']=df['created_at'].apply(lambda x:x.hour)
df['day']=df['created_at'].apply(lambda x:x.day)
df['month']=df['created_at'].apply(lambda x:x.month)
df['year']=df['created_at'].apply(lambda x:x.year)

df_grouped=df.groupby(['hour','day','month','year'])
"""
df_grouped=df.groupby(['hour','day','month','year'])

inspect=df_grouped.apply(lambda x:x.isnull().sum())

inspect_2=df.groupby(['hour','day','month','year'])['flag'].sum()


inspect_3=df.groupby(['hour','day','month','year'])['customer_id'].nunique()




#To Drop Duplicate Customers
df.drop_duplicates(subset=['customer_id'])


#Dropping duplicates for each group (Hourly Timestamp group)
inspect4=df_grouped.apply(lambda x:x.drop_duplicates(subset=['customer_id']))


"""

customer_grouped=df.groupby(['customer_id'])
p=[]
s=time.time()
def trip_auth(x):
  if len(x)>1: 
       x.sort_values(['created_at']) 
       x['shift_created_at']=x['created_at'].shift()
       x['shift_created_at'].iloc[0]=x['created_at'].iloc[1]-timedelta(hours=9)
       x['diff']=x.apply(lambda x:(x['created_at']-x['shift_created_at']).seconds,axis=1)
     
        
       x['auth_trip']=x['diff'].apply(lambda x:1 if x>=900.0 else 0) 
       x.reset_index(inplace=True)
       return x
  else:
       x.sort_values(['created_at']) 
       x['shift_created_at']=x['created_at'].shift()
       x['shift_created_at'].iloc[0]=x['created_at'].iloc[0]-timedelta(hours=9)
       x['diff']=x.apply(lambda x:(x['created_at']-x['shift_created_at']).seconds,axis=1)
     
        
       x['auth_trip']=x['diff'].apply(lambda x:1 if x>=900.0 else 0) 
       x.reset_index(inplace=True)
       return x
     
       
       
"""

Our Primary aim here is to find out the average false count for each customer in hourly timestamp. This can be used
as the value for unknown customer_ids i.e if the total unknown customer_ids in an hour is 20 and the 
average false count is 3 then the no of unique customers in that hour would be 20/(3+1)

"""


#Considering records which are only at least 15 minutes apart (For each customer)        
inspect5=customer_grouped.apply(trip_auth)   
print(time.time()-s)

#inspect5.dropna(inplace=True)

#inspect5.drop(index='customer_id')

inspect5.reset_index(drop=True,inplace=True)       
#inspect5.drop('customer_id',axis=1)
#inspect5.reset_index(inplace=True)
#inspect5=inspect5.rename(columns = {"index": "customer_id"})
   
#inspect6=inspect

customer_grouped2=inspect5.groupby(['customer_id'])


def total_counts(x):
    group=x.groupby(['hour','day','month','year'])['auth_trip'].count()
    return group


def mean_repeat(x):
    group=x.groupby(['hour','day','month','year'])['auth_trip'].sum()
    return group
    
s=time.time()
inspect6=customer_grouped2.apply(total_counts)



inspect7=customer_grouped2.apply(mean_repeat)
print(time.time()-s)



inspect6=inspect6.to_frame()
inspect7=inspect7.to_frame()

inspect6.columns=['total_counts']
inspect7.columns=['auth_sum']

inspect6.reset_index(inplace=True)
inspect7.reset_index(inplace=True)


inspect8=pd.merge(inspect6,inspect7,how='inner',on=['customer_id','hour','day','month','year'])
inspect8['false']=inspect8.apply(lambda x:1 if x['total_counts']>x['auth_sum'] else 0,axis=1) 

inspect8['false_count']=inspect8.apply(lambda x:x['total_counts']-x['auth_sum'],axis=1) 


"""

For each hour, false count mean will be different

"""

false_count_mean=inspect8.groupby(['hour','day','month','year'])['false_count'].mean()

false_count_mean=false_count_mean.to_frame()
false_count_mean.columns=['fc_mean']
false_count_mean.reset_index(inplace=True)


inspect9=pd.merge(false_count_mean,inspect8,how='inner',on=['hour','day','month','year'])



"""

Working on actual data

"""

DF=pd.merge(df,false_count_mean,how='inner',on=['hour','day','month','year'])

DF['customer_id'].fillna(-1,inplace=True)


DF['unknown_counts']=DF['customer_id'].apply(lambda x:1 if x==-1 else 0)



c_g=DF.groupby(['customer_id'])
p=[]
s=time.time()
def trip_auth2(x):
  if x['unknown_counts'].mean()!=1:  
   if len(x)>1: 
       x.sort_values(['created_at']) 
       x['shift_created_at']=x['created_at'].shift()
       x['shift_created_at'].iloc[0]=x['created_at'].iloc[1]-timedelta(hours=9)
       x['diff']=x.apply(lambda x:(x['created_at']-x['shift_created_at']).seconds,axis=1)
     
        
       x['auth_trip']=x['diff'].apply(lambda x:1 if x>=900.0 else 0) 
       x.reset_index(inplace=True)
       return x
   else:
       x.sort_values(['created_at']) 
       x['shift_created_at']=x['created_at'].shift()
       x['shift_created_at'].iloc[0]=x['created_at'].iloc[0]-timedelta(hours=9)
       x['diff']=x.apply(lambda x:(x['created_at']-x['shift_created_at']).seconds,axis=1)
     
        
       x['auth_trip']=x['diff'].apply(lambda x:1 if x>=900.0 else 0) 
       x.reset_index(inplace=True)
       return x
  else:
       x.sort_values(['created_at']) 
       x['shift_created_at']=x['created_at'].shift()
       x['shift_created_at'].iloc[0]=x['created_at'].iloc[0]-timedelta(hours=9)
       x['diff']=x.apply(lambda x:(x['created_at']-x['shift_created_at']).seconds,axis=1)
     
        
       x['auth_trip']=x['diff'].apply(lambda x:1) 
       x.reset_index(inplace=True)
       return x
     
#Considering records which are only at least 15 minutes apart (For each customer)        
inspect10=c_g.apply(trip_auth2)   
print(time.time()-s)
inspect10.reset_index(drop=True,inplace=True)

inspect10['auth_include']=inspect10.apply(lambda x:1 if x['unknown_counts']==0 else 0,axis=1)

def fn(x):
    if (x['auth_trip']==1)&(x['unknown_counts']==0):
        return 1 
    elif (x['auth_trip']==1)&(x['unknown_counts']==1):
        return 2
    else:
        return 0
    
inspect10['tot']=inspect10.apply(fn,axis=1)

known_trips=inspect10[inspect10['tot']==1]

unknown_trips=inspect10[inspect10['tot']==2]

known_trips[['created_at','lat','lng']].to_csv('known_trips.csv',index=False)

unknown_trips[['created_at','lat','lng']].to_csv('unknown_trips.csv',index=False)


unknown_trips[['created_at','lat','lng','fc_mean']].to_csv('unknown_trips_fc_mean.csv',index=False)


inspect10.to_csv('data.csv',index=False)

inspect11=inspect10[(inspect10['auth_trip']==1) | (inspect10['unknown_counts']==1)]

Final=f_customer_grouping=inspect10.groupby(['hour','day','month','year']).agg({'auth_include':sum,'unknown_counts':sum,'fc_mean':'mean'})     

Final['normalized_unknown_counts']=Final.apply(lambda x:x['unknown_counts']/(1+x['fc_mean']),axis=1)

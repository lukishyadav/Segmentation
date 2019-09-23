#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:13:21 2019

@author: lukishyadav
"""
import pandas as pd

#df=pd.read_csv('Supply_Data.csv')
import os
import time
start_time=time.time()
import sys
import datetime
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/‎Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply/darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_quantile_4_daily.csv')



dfs=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply/darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_quantile_4_hourly.csv')


dfd=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/data_big/data/quadrant_0/timescale_30/hex_edge_461.355m_quantile_4_hourly.csv')



DF=pd.merge(dfs,dfd,on='timeseries',how='inner')


L=dfs.columns

L[1]

%matplotlib auto
from matplotlib import pyplot as plt

plt.plot(DF[L[1]+'_x'],label='Supply')

plt.plot(DF[L[1]+'_y'],label='Demand')

plt.legend()

plt.plot()





file='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/data_big/data/quadrant_0/timescale_30/hex_edge_461.355m_quantile_4_hourly.csv'


df=pd.read_csv(file)

#Supply_Demand_Data_Sync
dfs=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply/darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_quantile_4_hourly.csv')

df=pd.merge(df,dfs['timeseries'],on='timeseries',how='inner')

DFS=pd.merge(dfs,df['timeseries'],on='timeseries',how='inner')



def quarter(df):
    #import re
    #result = re.search('ge_(.*)_hourly', file)
    #print(result.group(1))
    
    #qua=result.group(1)
    
    store=df.columns
    #mname=file[-35:-4]
    
    
    LL=len(df.columns)
    
    LLL=[str(i) for i in range((LL-1))]
    
    df.columns=['date']+LLL
    
    key=len(LLL)-2
    key=0
    #os.mkdir('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/images/'+result.group(1)+'_key_'+str(key)+'_'+store[key+1][1:-1])
    
    #dpath='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/images/'+result.group(1)+'_key_'+str(key)+'_'+store[key+1][1:-1]
    metric='hours'
    agg=6
    
    def convert(x):
        if metric=='hours':
            import re 
            import datetime
          
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
            return datetime.datetime(tup[0],tup[1],tup[2],tup[3])
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
    
    return data
    



ddd=quarter(df)


dds=quarter(DFS)

from matplotlib import pyplot as plt
plt.plot(ddd[43:48].values,label='Demand')
plt.plot(dds[43:48].values,label='Supply')
plt.legend()


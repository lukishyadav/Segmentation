#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 17:47:52 2019

@author: lukishyadav
"""
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
import my_module

df=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Commuters/commuters.csv')

df.isnull().sum()

df.dropna(inplace=True)

"""

['customer_id', 'rental_id', 'start location lat/lng',
       'end location lat/lng', 'Time taken from start to End', 'start time',
       'end datetime', 'Distance driven', 's_try', 's_lat_col', 's_lng_col',
       's_merc_lng', 's_merc_lat', 'e_try', 'e_lat_col', 'e_lng_col',
       'e_merc_lng', 'e_merc_lat']

"""

LAME=df.head(1000).copy()

DD=df.groupby(['customer_id']).size().reset_index(name='counts')

CS=list(set(DD['customer_id'][DD['counts']>4]))

global CUSTOMER
CUSTOMER=33569

#95 gives decent results

def convert_to_mercator(lngs, lats):
    projection = Proj(init='epsg:3857')
    xs = []
    ys = []
    for lng, lat in zip(lngs, lats):
        x, y = projection(lng, lat)
        xs.append(x)
        ys.append(y)
    return xs, ys


df['s_try']=df['start location lat/lng'].apply(lambda x:x.split(","))

df['s_lat_col']=df['s_try'].apply(lambda x:float(x[0]))

df['s_lng_col']=df['s_try'].apply(lambda x:float(x[1]))


df['s_merc_lng'],df['s_merc_lat']=convert_to_mercator(df['s_lng_col'],df['s_lat_col'])



df['e_try']=df['end location lat/lng'].apply(lambda x:x.split(","))

df['e_lat_col']=df['e_try'].apply(lambda x:float(x[0]))

df['e_lng_col']=df['e_try'].apply(lambda x:float(x[1]))


df['e_merc_lng'],df['e_merc_lat']=convert_to_mercator(df['e_lng_col'],df['e_lat_col'])


maxlat=max(df['s_merc_lat'])
minlat=min(df['s_merc_lat'])

maxlng=max(df['s_merc_lng'])
minlng=min(df['s_merc_lng'])






kms_per_radian = 6371.0088


global min_meters 

def set_clusters(df, percent, min_n, min_meters):
    # set the data up for clustering
    min_percent = int(df.shape[0]/100) * percent  # 1% of samples available
    epsilon = float(min_meters)/1000/kms_per_radian  # m * (1km/1000m) * (radian/km) => radians
    X = np.column_stack((df['s_lng_col'], df['s_lat_col']))
    e_X = np.column_stack((df['e_lng_col'], df['e_lat_col']))
    # clustering
    dbscan = DBSCAN(eps=epsilon, min_samples=max(min_percent, min_n),
                    algorithm='ball_tree',
                    metric='haversine').fit(np.radians(X))
    
    e_dbscan = DBSCAN(eps=epsilon, min_samples=max(min_percent, min_n),
                    algorithm='ball_tree',
                    metric='haversine').fit(np.radians(e_X))

    # create list of clusters and labels
    #n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    #n_noise_ = list(dbscan.labels_).count(-1)

    # label the points unique to their cluster
    #unique_labels = set(dbscan.labels_)
    df['label'] = [str(label) for label in dbscan.labels_]
    df['e_label'] = [str(label) for label in e_dbscan.labels_]

    return df


def set_time_clusters(df,min_n, epsilon):
    # set the data up for clustering
    X = np.column_stack((df['seconds_start'], df['seconds_end']))
    # clustering
    dbscan = DBSCAN(eps=epsilon, min_samples=min_n,).fit(np.radians(X))
    # create list of clusters and labels
    #n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    #n_noise_ = list(dbscan.labels_).count(-1)

    # label the points unique to their cluster
    #unique_labels = set(dbscan.labels_)
    df['time_label'] = [str(label) for label in dbscan.labels_]


    return df


min_meters=50

"""
minPts = 2·dim can be used
minPts=4 here... (minimum)

"""
#runfile('C:/yourfolder/myfile.py',args='one two three')


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

"""
import itertools

sliced = list(itertools.islice(CS, 1))

#global DF

DF=df[df['customer_id'].isin(sliced)]
DF['label']=np.zeros(len(DF))
DF['e_label']=np.zeros(len(DF))

import time
s=time.time()

for x in sliced:
    #CUSTOMER=x
    
    DF.loc[DF['customer_id']==x]=set_clusters(DF[DF['customer_id']==x],5,5,100)

print(time.time()-s)                  

"""

#DF.to_csv('/Users/lukishyadav/Desktop/Gittable_Work/Commuters/100_Customers_CLustered.csv',index=False)
#DF=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Commuters/100_Customers_CLustered.csv')

"""
import itertools

sliced = list(itertools.islice(CS, 10000))

sliced2 = list(itertools.islice(CS, 10000))

#global DF

DF=df[df['customer_id'].isin(sliced)]
DF['label']=np.zeros(len(DF))
DF['e_label']=np.zeros(len(DF))
"""

df=df[df['customer_id'].isin(CS)]

XXX=df.groupby('customer_id')

import time
s=time.time()

g=XXX.apply(lambda x:set_clusters(x,5,5,100))

print(time.time()-s)  



"""

3491.7077510356903 seconds taken for 1000 records

"""





#DF.to_csv('/Users/lukishyadav/Desktop/Gittable_Work/Commuters/100_Customers_CLustered.csv',index=False)
 
"""
We have the label and e_label in dataframe


records having both label and e_label are rentals with commute between clusters i.e rental counted as commuters rental.


"""

#len(DF)

"""
Commuters rentals
"""





DF=g.copy()

#DF=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Commuters/1000_Customers_CLustered')

final_count=DF.groupby(['customer_id']).size().reset_index(name='counts')

#DF2=DF[DF['customer_id']==95].copy()
#DF['label'].iloc[1]

DF2=DF.copy()

DDD=DF2[(DF2['label']!="-1") & (DF2['e_label']!="-1")]

final_count2=DDD.groupby(['customer_id']).size().reset_index(name='counts')

Commuters=pd.merge(final_count,final_count2, on='customer_id', how='left')
Commuters.fillna(0,inplace=True)

Commuters['factor']=Commuters[['counts_x','counts_y']].apply(lambda x:float(x[1]/x[0]),axis=1)

Commuters=Commuters[Commuters['factor']!=0]

"""

Zipped=list(zip(DDD['label'],DDD['e_label']))

zipped_set=set(Zipped)

import collections
collections.Counter(Zipped)

#s1.line([2,3],[4,6],line_width=2)

lngs=list(zip(DDD['s_merc_lng'],DDD['e_merc_lng']))
lats=list(zip(DDD['s_merc_lat'],DDD['e_merc_lat']))

dataf=pd.DataFrame()
dataf['lats']=lats
dataf['lngs']=lngs
dataf['Zipped']=Zipped

zipped_set

"""

#m,n,factor=my_module.remove_outliers(Commuters,'factor') 

m,n,counts_x=my_module.remove_outliers(Commuters,'counts_x') 


#M,N,Factor=my_module.Remove_Outliers(Commuters,'factor',1) 

sns.boxplot(x=Factor['factor'])


finale=pd.merge(factor,counts_x, on='customer_id', how='inner')




m,n,counts_x=my_module.remove_outliers(Commuters,'counts_x') 

finale=pd.merge(Commuters,counts_x, on='customer_id', how='inner')

finale.reset_index(inplace=True)

%matplotlib auto
X=finale[['factor_x','counts_x_y']].values
from sklearn.preprocessing import MinMaxScaler
M=MinMaxScaler(feature_range=(0, 1), copy=True)
M.fit(X)
xX=M.transform(X)
#xX=X`
 
from sklearn.cluster import KMeans
algo=KMeans(n_clusters=6, random_state=0)
h=my_module.Cluster(xX,algo,x_l='Common_to_Total_rental_ratio',y_l='Rental_Counts')


from sklearn.cluster import AgglomerativeClustering
algo = AgglomerativeClustering(n_clusters=4).fit(X)
h=my_module.Cluster(xX,algo,x_l='Common_to_Total_rental_ratio',y_l='Rental_Counts')



List=list(h)

LIST=[index for index, value in enumerate(List) if value == 1]

Check=finale.ix[LIST]  



Check3=Check[['counts_x_y','factor_x']]

Check2=Check[['counts_x_y','factor_x']]


Check2.to_csv('/Users/lukishyadav/Desktop/Gittable_Work/Commuters/mid_Commuters.csv',index=False)

Check3.to_csv('/Users/lukishyadav/Desktop/Gittable_Work/Commuters/low_high_Commuters.csv',index=False)


"""

Clustering on the  basis of start and endtime of rental for a customer.


['customer_id', 'rental_id', 'start location lat/lng',
       'end location lat/lng', 'Time taken from start to End', 'start time',
       'end datetime', 'Distance driven']



'start time','end datetime'
       
"""

lame=df.head(1000).copy()
lame=df[df['customer_id']==33569].copy()

lame['start time'].iloc[1]

fmt = '%Y-%m-%d %H:%M:%S'
f='%Y-%m-%d'
then = datetime.datetime.strptime('2018-12-31 23:59:59', f)


x='2018-12-31 23:59:59'


"""
Function to convert timestamp in to no of seconds since day started

"""

fmt = '%Y-%m-%d %H:%M:%S'
f='%Y-%m-%d'


def in_seconds(x):
    current = datetime.datetime.strptime(x[0:19], fmt)
    day=x[0:10]
    Day=datetime.datetime.strptime(day, f)
    value=(current-Day).seconds
    return value

lame['seconds_start']=lame['start time'].apply(in_seconds)

lame['seconds_end']=lame['end datetime'].apply(in_seconds)


df['start time'].iloc[1]

df['seconds_start']=df['start time'].apply(in_seconds)

df['seconds_end']=df['end datetime'].apply(in_seconds)
    
C=df.describe() 

df.dtypes

df.iloc[df.dtypes == float,5]
 
df.select_dtypes(include ='float64')
   
df.isnull().sum()

df.dropna(inplace=True)

# =============================================================================
# df['Days from last rental'].iloc[1][0:19]
# 
# df['Days from last rental']=df['Days from last rental'].apply(lambda x:datetime.datetime.strptime(x[0:19], fmt))
# 
# 
# df['Days from last rental']=df['Days from last rental'].apply(lambda x:then-x)
# 
# 
# df['Days from last rental']=pd.to_timedelta(df['Days from last rental'])
# =============================================================================


xX=lame[['seconds_start','seconds_end']].values

%matplotlib auto
from sklearn.cluster import DBSCAN
ms=len(xX)*0.05
algo = DBSCAN(eps=1200, min_samples=ms).fit(xX)
h=my_module.Cluster(xX,algo,x_l='Common_to_Total_rental_ratio',y_l='Rental_Counts')





from sklearn.cluster import OPTICS

algo = OPTICS(min_samples=5, xi=.05, min_cluster_size=.05,max_eps=1000)
h=my_module.Cluster(xX,algo,x_l='Common_to_Total_rental_ratio',y_l='Rental_Counts')


lame.reset_index(inplace=True)

List=list(h)

LIST=[index for index, value in enumerate(List) if value != -1]

Check=lame.ix[LIST]  



import time
s=time.time()

G=XXX.apply(lambda x:set_time_clusters(x,5,1200))

print(time.time()-s)  




FD=G.copy()

#DF=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Commuters/1000_Customers_CLustered')



final_count=FD.groupby(['customer_id']).size().reset_index(name='counts')

#DF2=DF[DF['customer_id']==95].copy()






FD2=FD.copy()

DDD=FD2[(FD2['time_label']!="-  1")]

final_count2=DDD.groupby(['customer_id']).size().reset_index(name='counts')


T_Commuters=pd.merge(final_count,final_count2, on='customer_id', how='left')
T_Commuters.fillna(0,inplace=True)

T_Commuters['factor']=T_Commuters[['counts_x','counts_y']].apply(lambda x:float(x[1]/x[0]),axis=1)

T_Commuters=T_Commuters[T_Commuters['factor']!=0]



Def=pd.merge(T_Commuters[['customer_id','factor']],Commuters[['customer_id','factor']],on='customer_id',how='inner')
Def.columns=['customer_id','time_factor','factor']

# =============================================================================
# %matplotlib auto
# X=finale[['factor_x','counts_x_y']].values
# from sklearn.preprocessing import MinMaxScaler
# M=MinMaxScaler(feature_range=(0, 1), copy=True)
# M.fit(X)
# xX=M.transform(X)
# =============================================================================
#xX=X`
xX=Def[['time_factor','factor']].values


%matplotlib auto
from sklearn.cluster import KMeans
algo=KMeans(n_clusters=6, random_state=0)
h=my_module.Cluster(xX,algo,x_l='time_factor',y_l='factor')


from sklearn.cluster import AgglomerativeClustering
algo = AgglomerativeClustering(n_clusters=4)
h=my_module.Cluster(xX,algo,x_l='time_factor',y_l='factor')



fds=g[g['customer_id']==33569]
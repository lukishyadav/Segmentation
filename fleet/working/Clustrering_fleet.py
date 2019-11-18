#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:45:20 2019

@author: lukishyadav
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:10:06 2019

@author: lukishyadav


Clustering???

(DBscan, H???)

one hot encoding of zippys?  (end rentals)

rentals since last cleaning???


checking out on more generalized models????

"""

import pandas as pd
import datetime
import statistics
from numpy import mean
import collections

from settings import region


REGION_TIMEZONE = region['oakland']['timezone']

def convert_datetime_columns(df, columns):
    for col in columns:
        try:
            df[col] = df[col].dt.tz_localize('UTC').dt.tz_convert(REGION_TIMEZONE)
        except TypeError:
            df[col] = df[col].dt.tz_convert(
                   'UTC').dt.tz_convert(REGION_TIMEZONE)


df=pd.read_csv('ldata.csv')


df.columns=['tenant_id','start_address','vehicle_id','end_address', 'timestamp', 'rental_started_at',
       'rental_ended_at', 'name', 'rental_id',
       'fuel_percent_start', 'fuel_percent_end',
       'Fuel Used', 'distance driven','rating_reason', 'rating_score']


df.drop_duplicates(['rental_id'],keep= 'last',inplace=True)

df['rating_reason'].fillna(0,inplace=True)


df.dropna(inplace=True)

#fd=df.copy()
#fd.dropna(inplace=True)

VEHICLE_DT_COLS=['rental_started_at','rental_ended_at']


df=df[(df['distance driven']>0) & (df['Fuel Used']>0)]


df['rental_started_og']=df['rental_started_at']
df['rental_started_at']=df['rental_started_at'].apply(lambda x:datetime.datetime.strptime(x[0:19], '%Y-%m-%dT%H:%M:%S'))

df['rental_ended_at']=df['rental_ended_at'].apply(lambda x:datetime.datetime.strptime(x[0:19], '%Y-%m-%dT%H:%M:%S'))

df['rental_duration']=df.apply(lambda x:x['rental_ended_at']-x['rental_started_at'],axis=1)

df['hour']=df['rental_ended_at'].apply(lambda x:x.hour)

df['weekday']=df['rental_ended_at'].apply(lambda x:x.weekday())

df['duration_hours']=df['rental_duration'].apply(lambda x:(x.total_seconds()/3600))

df['Fuel Used']=df.apply(lambda x:float(x['fuel_percent_start']-x['fuel_percent_end']),axis=1)

import re

def zip(x):
    if len(re.findall(r'[0-9][0-9][0-9][0-9][0-9]', x))>0:
        return re.findall(r'[0-9][0-9][0-9][0-9][0-9]', x)[0]
    else:
        return 'missing'


df['start_zip']=df['start_address'].apply(zip)
df['end_zip']=df['end_address'].apply(zip)



convert_datetime_columns(df, VEHICLE_DT_COLS)



def outlier_trimmean(arr, percent):
    n = len(arr)
    k = int(round(n*(float(percent)/100)/2))
    tmean=mean(arr[k+1:n-k])
    tstd=statistics.stdev(arr[k+1:n-k])
    return arr[(arr>(tmean-3*tstd)) & (arr<(tmean+3*tstd))]


def outlier_trimmean(DF,c, percent):
    arr=DF[c]
    n = len(arr)
    k = int(round(n*(float(percent)/100)/2))
    tmean=mean(arr[k+1:n-k])
    tstd=statistics.stdev(arr[k+1:n-k])
    return DF[(DF[c]>(tmean-3*tstd)) & (DF[c]<(tmean+3*tstd))]

DF=outlier_trimmean(df,'distance driven',5)

DF=outlier_trimmean(DF,'distance driven',5)


DF=DF[(DF['end_zip']>'10000') & (DF['end_zip']!='missing')]

#DF=DF[(DF['end_zip']!='missing')]


DF['ez']=DF['end_zip'].apply(lambda x:eval(x))

DF=outlier_trimmean(DF,'ez',5)


DF['ratings']=DF['rating_score'].apply(lambda x:1 if x>3.0 else 0)

collections.Counter(DF['ratings'])



"""

#Including negative ratings w.r.t cleanlinesss only

cDF=DF[((DF['rating_reason']=='Cleanliness') & (DF['ratings']==0)) | (DF['ratings']==1)]

clean_df=DF[(DF['rating_reason']=='Cleanliness')]

features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez']
label=['ratings']
#label='rating_score'

X=DF[features]
y=DF['ratings']

DF['rental_ended_at']

"""



"""
import datetime
print (datetime.date.today() - datetime.timedelta(6*365/12))   .isoformat()

dtt=pd.DataFrame(data=[datetime(2019, 4,1)],columns=['date'])

#from datetime import datetime
dtt=pd.DataFrame(data=[pd.Timestamp(datetime.date.today() - datetime.timedelta(6*365/12))],columns=['date'])

dtt['date']=dtt['date'].dt.tz_localize('UTC').dt.tz_convert(REGION_TIMEZONE)

DF['month']=DF['rental_ended_at'].apply(lambda x:x.month)

DF['year']=DF['rental_ended_at'].apply(lambda x:x.year)

#DF['myear']=DF.apply(lambda x:str(x['month'])+'-'+str(x['year']),axis=1)

"""

DF['myear']=DF.apply(lambda x:datetime.datetime.strptime(str(x['rental_ended_at'])[0:7], '%Y-%m'),axis=1)





DF['ratings']=DF['rating_score'].apply(lambda x:1 if x>3.0 else 0)

collections.Counter(DF['ratings'])

"""
Pickling to save time and avoid repeated runs

"""


import pickle
dbfile = open('processed_dataframe', 'ab') 
      
# source, destination 
pickle.dump(DF, dbfile)                      
dbfile.close() 
  

dbfile = open('processed_dataframe', 'rb')      
DF = pickle.load(dbfile) 
    
    
DF3=DF.copy()



features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez']
label=['ratings']


X=DF[features]
y=DF['ratings']



DF=DF[DF['rating_score']!=5.0]


features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez']
label=['ratings']
#label='rating_score'

#X=DFtrain[features]
#y=DFtrain[label]

"""
DF=DF3.copy()
DF=DF[DF['myear']==MD[-3]]


DF=DF[DF['rating_score']!=5.0]

"""
X=DF[features]
y=DF['ratings']






"""
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

#os = SMOTE(random_state=0)


us=RandomUnderSampler(random_state=0)


ux,uy=us.fit_sample(X,y)


columns=features.copy()
columns.extend(label)
FD=DF[columns].head(len(uy))

FD['ratings']=uy
FD[features]=ux


"""
X=FD[features]
y=FD['ratings']


#Clustering

import my_module
import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
 
pca.fit(X)  


pX=pca.transform(X)


print(pca.explained_variance_ratio_)  

from sklearn.cluster import KMeans
algo=KMeans(n_clusters=10)

clus=my_module.Cluster(pX,algo,x_l='x',y_l='y')


DF['cluster']=list(clus)


final=DF[['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez','ratings','cluster']]


features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez']

features=['Fuel Used','distance driven']

X=DF[features]
clus=my_module.Cluster(X.values,algo,x_l='x',y_l='y')


X['label']=clus

features
features2=features

features2.pop(1)

X=PH[features2]
y=PH['ratings']

features2.extend(['ratings'])

features2.extend(['rating_score'])

CC=PH[features2]

CHECK=CC.corr()

CC2=CC[CC['rating_score']!=5.0]

CHECK2=CC.corr()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
sX=scaler.fit_transform(X)
"""

import numpy as np


X.values[1000,:]


u,w,v = np.linalg.svd(sX[0:1000,:])

v[0,:]

g=v[1,:]


from matplotlib import pyplot as plt
plt.plot(v[6,:])

plt.scatter(v[0,:],v[1,:])
plt.xlabel('V0')
plt.ylabel('V1')


plt.scatter(v[0,:],v[2,:])
plt.xlabel('V0')
plt.ylabel('V2')


plt.scatter(v[1,:],v[2,:])
plt.xlabel('V1')
plt.ylabel('V2')

g.T*g

np.inner( g ,g )

u, s, vh = np.linalg.svd(a, full_matrices=False)


X-- getting rid of

fuel used


could be good features::
    
distance driven

duration_hours

----- To Do -------

documentation (everything to be included)

different scalers

looking at label encoders of zip codes


simple models vs complex




    
"""

data=X

import seaborn as sns

facet = sns.lmplot(data=data, x='Fuel Used', y='distance driven', hue='label', 
                   fit_reg=False, legend=True, legend_out=True)



facet = sns.lmplot(data=data, x='Fuel Used', y='distance driven', hue='label', 
                   fit_reg=False, legend=True, legend_out=True)



km = algo.fit(X)
Clusters=km.labels_

DF['cluster']=list(Clusters)

final=DF[['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez','ratings','cluster']]





from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=3, min_samples=2).fit(X)

clusters=clustering.labels_

DF['cluster']=list(clusters)

finaldbscan=DF[['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez','ratings','cluster']]


features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez']



X=DF[features]



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

XX=scaler.fit_transform(X)



""" Kmeans """
from sklearn.cluster import KMeans
algo=KMeans(n_clusters=2)

km = algo.fit(XX)
Clusters=km.labels_

PH['cluster']=list(Clusters)

final=PH[['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez','ratings','num','cluster']]



from sklearn.cluster import AgglomerativeClustering


clustering = AgglomerativeClustering(n_clusters=2).fit(XX)
Clusters=clustering.labels_
DF['cluster']=list(Clusters)

finalagglon=DF[['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez','ratings','cluster']]




from sklearn.cluster import MeanShift

clustering = MeanShift(bandwidth=2).fit(XX)

Clusters=clustering.labels_
DF['cluster']=list(Clusters)

finalmeanshift=DF[['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez','ratings','cluster']]



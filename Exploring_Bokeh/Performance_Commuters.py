#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:33:52 2019

@author: lukishyadav
"""

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



def p_set_clusters(df, percent, min_n, min_meters):
    # set the data up for clustering
    min_percent = int(df.shape[0]/100) * percent  # 1% of samples available
    epsilon = float(min_meters)/1000/kms_per_radian  # m * (1km/1000m) * (radian/km) => radians
    X = np.column_stack((df['s_lng_col'], df['s_lat_col']))
    e_X = np.column_stack((df['e_lng_col'], df['e_lat_col']))
    # clustering
    dbscan = DBSCAN(eps=epsilon, min_samples=max(min_percent, min_n),
                    algorithm='ball_tree',
                    metric='haversine',n_jobs=-1).fit(np.radians(X))
    
    e_dbscan = DBSCAN(eps=epsilon, min_samples=max(min_percent, min_n),
                    algorithm='ball_tree',
                    metric='haversine',n_jobs=-1).fit(np.radians(e_X))

    # create list of clusters and labels
    #n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    #n_noise_ = list(dbscan.labels_).count(-1)

    # label the points unique to their cluster
    #unique_labels = set(dbscan.labels_)
    df['label'] = [str(label) for label in dbscan.labels_]
    df['e_label'] = [str(label) for label in e_dbscan.labels_]

    return df



import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import itertools

sliced = list(itertools.islice(CS, 100))

#global DF

DF=df[df['customer_id'].isin(sliced)]
DF['label']=np.zeros(len(DF))
DF['e_label']=np.zeros(len(DF))


XXX=DF.groupby('customer_id')

g=XXX.apply(lambda x:set_clusters(x,5,5,100))



DF=DF[DF['customer_id']==int(21)].copy()


sliced=[21]


import time
s=time.time()

for x in sliced:
    #CUSTOMER=x
    
    DF.loc[DF['customer_id']==x]=set_clusters(DF[DF['customer_id']==x],5,5,100)

print(time.time()-s)                  


import multiprocessing

multiprocessing.cpu_count()



"""

For Faster DBSCANing of customers

"""

import itertools

sliced = list(itertools.islice(CS, 100))

sliced2 = list(itertools.islice(CS, 100))

#global DF

DF=df[df['customer_id'].isin(sliced)]
DF['label']=np.zeros(len(DF))
DF['e_label']=np.zeros(len(DF))


XXX=DF.groupby('customer_id')

import time
s=time.time()

g=XXX.apply(lambda x:set_clusters(x,5,5,100))

print(time.time()-s)  


g['label']=g['label'].astype(float)
g['e_label']=g['e_label'].astype(float)



DF[g['label']!=DF['label']]

DF['label'].iloc[1]










XXX=DF.groupby('customer_id')

import time
s=time.time()

g=XXX.apply(lambda x:set_clusters(x,5,5,100))

print(time.time()-s) 


sliced = list(itertools.islice(CS, 100))

sliced2 = list(itertools.islice(CS, 100))

#global DF

df2=df[df['customer_id'].isin(sliced)]

import numpy as np

def greater(x):
    x['Distance driven']=x['Distance driven'].astype(float)
    x['distance']=np.where((x['Distance driven']>20.0),1,0)
    TH=len(x)
    x['count']=np.where(TH>0,TH,0)
    return x
    
XX=df2.groupby('customer_id')    

G=XX.apply(lambda x:greater(x))
 

df['Distance driven']=df['Distance driven'].astype(float)

df['distance']=np.where(1 if df['Distance driven']>20.0 else 0)




"""

MultiThreading Attempt 


"""


import time
s=time.time()


from multiprocessing import Pool

def f(x):
    XX=x.groupby('customer_id')  
    G=XX.apply(lambda x:greater(x))
    print(len(G))

if __name__ == '__main__':
    p = Pool(5)
    print(p.map(f, DF))
    
    
print(time.time()-s)     
    





from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    p = Pool(5)
    print(p.map(f, [1, 2, 3]))    
    
    
 
import itertools

sliced = list(itertools.islice(CS, 100))
sliced2 = list(itertools.islice(CS, 100))

#global DF
dummy=df[df['customer_id'].isin(sliced)] 

L=list(dummy.groupby('customer_id').groups.items())

# =============================================================================
# List=list(h)
# 
# LIST=[index for index, value in enumerate(List) if value == 1]
# 
# Check=Data.ix[LIST]  
# =============================================================================
 

def p_set_clusters(percent, min_n, min_meters,df):
        # set the data up for clustering
        min_percent = int(df.shape[0]/100) * percent  # 1% of samples available
        epsilon = float(min_meters)/1000/kms_per_radian  # m * (1km/1000m) * (radian/km) => radians
        X = np.column_stack((df['s_lng_col'], df['s_lat_col']))
        e_X = np.column_stack((df['e_lng_col'], df['e_lat_col']))
        # clustering
        dbscan = DBSCAN(eps=epsilon, min_samples=max(min_percent, min_n),
                        algorithm='ball_tree',
                        metric='haversine',n_jobs=-1).fit(np.radians(X))
        
        e_dbscan = DBSCAN(eps=epsilon, min_samples=max(min_percent, min_n),
                        algorithm='ball_tree',
                        metric='haversine',n_jobs=-1).fit(np.radians(e_X))
    
        # create list of clusters and labels
        #n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        #n_noise_ = list(dbscan.labels_).count(-1)
    
        # label the points unique to their cluster
        #unique_labels = set(dbscan.labels_)
        df['label'] = [str(label) for label in dbscan.labels_]
        df['e_label'] = [str(label) for label in e_dbscan.labels_]

        return df


import multiprocessing as mp
import pandas as pd
from functools import partial
  
pool = mp.Pool(processes = (multiprocessing.cpu_count()  - 1))
func = partial(p_set_clusters,df=dummy,percent=5,min_n=5,min_meters=100)
results = pool.map(func,L)
pool.close()
pool.join()

results_df = pd.concat(results)

import multiprocessing as mp

multiprocessing.cpu_count()    


type(L[1][1])


XXX=dummy.groupby('customer_id')



func = partial(p_set_clusters,percent=5,min_n=5,min_meters=100)
results = pool.map(func,L)

def applyParallel(dfGrouped, func):
    with Pool(multiprocessing.cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in XXX])
    return pd.concat(ret_list)






import time
s=time.time()

g=XXX.apply(lambda x:set_clusters(x,5,5,100))

print(time.time()-s)  




XXX=df.groupby('customer_id')

import time
s=time.time()

g=XXX.apply(lambda x:set_clusters(x,5,5,100))

print(time.time()-s)  






def p_set_clusters(df,percent, min_n, min_meters):
        # set the data up for clustering
        min_percent = int(df.shape[0]/100) * percent  # 1% of samples available
        epsilon = float(min_meters)/1000/kms_per_radian  # m * (1km/1000m) * (radian/km) => radians
        X = np.column_stack((df['s_lng_col'], df['s_lat_col']))
        e_X = np.column_stack((df['e_lng_col'], df['e_lat_col']))
        # clustering
        dbscan = DBSCAN(eps=epsilon, min_samples=max(min_percent, min_n),
                        algorithm='ball_tree',
                        metric='haversine',n_jobs=-1).fit(np.radians(X))
        
        e_dbscan = DBSCAN(eps=epsilon, min_samples=max(min_percent, min_n),
                        algorithm='ball_tree',
                        metric='haversine',n_jobs=-1).fit(np.radians(e_X))
    
        # create list of clusters and labels
        #n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        #n_noise_ = list(dbscan.labels_).count(-1)
    
        # label the points unique to their cluster
        #unique_labels = set(dbscan.labels_)
        df['label'] = [str(label) for label in dbscan.labels_]
        df['e_label'] = [str(label) for label in e_dbscan.labels_]

        return df



func = partial(p_set_clusters,percent=5,min_n=5,min_meters=100)


import time
s=time.time()

def parallelize_dataframe(df, func, n_cores=11):
    df_split = np.array_split(df, n_cores)
    pool = mp.Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


T=parallelize_dataframe(XXX,func)

print(time.time()-s)  






XXX=df.groupby('customer_id')



import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
import multiprocessing as mp
import pandas as pd
from functools import partial

import time
s=time.time()

func = partial(p_set_clusters,percent=5,min_n=5,min_meters=100)

from multiprocessing import Pool, cpu_count
import multiprocessing

def applyParallel(dfGrouped, func):
    with Pool(multiprocessing.cpu_count()-1) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)


T=applyParallel(XXX, func)

print(time.time()-s) 

import numpy as np
import pandas as pd
import time
s=time.time()


df=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Commuters/commuters.csv')
print(time.time()-s) 



import dask.dataframe as dd
s=time.time()
df = dd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Commuters/commuters.csv')
print(time.time()-s) 







kms_per_radian = 6371.0088

XXX=df.groupby('customer_id')

import time
s=time.time()

g=XXX.apply(lambda x:set_clusters(x,5,5,100))

print(time.time()-s)  








from functools import partial


kms_per_radian = 6371.0088
XXX=df.groupby('customer_id')

import time
s=time.time()

func = partial(p_set_clusters,percent=5,min_n=5,min_meters=100)

from multiprocessing import Pool, cpu_count

def applyParallel(dfGrouped, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pandas.concat(ret_list)

T=applyParallel(XXX, func)

print(time.time()-s) 

T.iloc[1]

print(1)


for name, group in XXX:
    print(name,group)

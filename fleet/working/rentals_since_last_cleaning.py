#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:47:05 2019

@author: lukishyadav
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

df['rating_reason'].fillna(0,inplace=True)


df.dropna(inplace=True)


df.drop_duplicates(['rental_id'],keep= 'last',inplace=True)
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


import pickle

dbfile = open('processed_dataframe', 'rb')      
DF = pickle.load(dbfile) 
    


#DF.drop_duplicates(['rental_id'],keep= 'last',inplace=True)

import pandas as pd
dd=pd.read_csv('DarwinJobssmartquery_modified_2019-11-15_0845.csv')


"""dd.columns=['id', 'created_at', 'updated_at', 'status', 'fleet_manager_comment',
       'supplier_comment_to_fleet_manager', 'supplier_comment_to_contractor',
       'contractor_comment', 'assigned_to_fleet_manager',
       'service_block_queued', 'due_date', 'status_updated_at', 'cancelled_at',
       'closed_at', 'completed_by_supplier_at', 'in_progress_at',
       'fleet_manager_priority', 'supplier_manager_priority',
       'service_block_id', 'supplier_company_id', 'supplier_contractor_id',
       'vehicle_id', 'service_id', 'afm_job_xid', 'afm_updated_at',
       'vehicle_alert_id', 'id.1', 'ticket_id', 'servicetype_id', 'id.2',
       'name', 'cost', 'created_at.1', 'updated_at.1', 'type']
"""


dd=pd.read_csv('DarwinCarJobs_2019-11-5_1504.csv')

dd=pd.read_csv('AllDarwinCarJobs_2019-11-14_1028.csv')

"""
['id', 'Worker First Name', 'Worker Last Name', 'Worker Email',
       'job type', 'priority', 'status', 'target_entity_primary_identifier',
       'target_entity_secondary_identifier', 'Creation Time PT',
       'Assignment Time PT', 'Start Time PT', 'End Time PT',
       'Cancelled Time PT', 'Expected Completion Time', 'Worker Time Increase',
       'Job Dispatch Time', 'Job Lead Time', 'Job Work Time', 'Job Total Time',
       'Delta Work Time', 'Delta Work Time with Worker Time Increase',
       'user_xid', 'service_xid']
"""

v=pd.read_csv('Vehiclelicenseplate_modified_2019-11-6_1228.csv')

v.columns=['vehicle_id','target_entity_secondary_identifier']


main_df=pd.merge(dd,v,on='target_entity_secondary_identifier',how='inner')




"""
top 20 vehicles to be considered?? No!!!

"""

import datetime

dv=dict(collections.Counter(main_df['vehicle_id']))
import operator
sorted_dv = sorted(dv.items(), key=operator.itemgetter(1),reverse=True)

sorted_dv=dict(sorted_dv[0:20])

main_df_filtered=main_df[main_df['job type'].isin(['14-Day Cleaning BA','Requested Cleaning - SAC','Requested Cleaning BA'])]

MF=main_df_filtered[main_df_filtered['vehicle_id']==329]

main_df_filtered['End Time PT']=main_df_filtered['End Time PT'].apply(lambda x:datetime.datetime.strptime(x[0:19], '%Y-%m-%d %H:%M:%S'))



DT_COLS=['End Time PT']

"""
Converting to American Time from UTC
"""

convert_datetime_columns(main_df_filtered, DT_COLS)


"""
Filtering out deleted workers
"""


main_df_filtered2=main_df_filtered[main_df_filtered['Worker First Name']!='DELETED']


"""
Filtering out columns
"""

df2=main_df_filtered2[['End Time PT','vehicle_id','job type']]


df2=main_df_filtered2[['End Time PT','vehicle_id','job type','tenant_xid']]



#group=DF.groupby(['vehicle_id'])


import datetime 
now = datetime.datetime.now()

from datetime import datetime

"""
Taking threshold date and converting it to american time. 
Threshold date is required since our AFM data is available only from threshold date.
"""

dtt=pd.DataFrame(data=[datetime(2019, 4,1)],columns=['date'])

dtt['date']=dtt['date'].dt.tz_localize('UTC').dt.tz_convert(REGION_TIMEZONE)

DF2=DF[DF['rental_ended_at']>dtt['date'].iloc[0]]



DF2=DF2[DF2['vehicle_id'].isin(list(sorted_dv.keys()))]
"""
Creating groupbys
"""

group=DF2.groupby(['vehicle_id'])

DF329.sort_values(by=['vehicle_id'])

import time
s=time.time()

mgroup=pd.DataFrame()

def days_from_last_cleaning(x):
    
    global mgroup
    
    #print(x.columns)
    
    x.sort_values(by=['rental_started_at'],inplace=True)
    
    O=df2[df2['vehicle_id']==max(x['vehicle_id'])]
    
    O.sort_values(by=['End Time PT'],inplace=True)
    
    O['rental_started_at']=O['End Time PT']
        
    xyz=pd.merge_asof(x, O, on='rental_started_at')
    mgroup=mgroup.append(xyz)
    
        
    
    



group.apply(days_from_last_cleaning)

mgroup2=mgroup.dropna()

print(time.time()-s)



"""


mgroup3=mgroup[mgroup['vehicle_id_x']==514]

mgroup3['num']=['a' for n in range(len(mgroup3))]

mylist=list(set(mgroup3['End Time PT']))

subgroup=mgroup3.groupby(['End Time PT'])

"""
calculating rental since last cleaning
"""
x=mgroup3[mgroup3['End Time PT']==mylist[-1]]

outcome=pd.DataFrame()

def subgroupf(x):
    x.sort_values(by=['rental_started_at'],inplace=True)
    L=list((range(len(x))))
    x['num']=L
    print(x.columns)
    outcome.append(x)
    
    """
    ph=mgroup3[mgroup3['End Time PT']==max(x['End Time PT'])]
    ph.sort_values(by=['rental_started_at'],inplace=True)
    ph['num'][mgroup3['End Time PT']==x['End Time PT'].iloc[0]]=L
    outcome=outcome.append(ph)
    """
subgroup.apply(subgroupf)    


"""


PH=pd.DataFrame()    

for xx in list(set(mgroup['vehicle_id_x'])):
    mgroup4=mgroup2[mgroup2['vehicle_id_x']==xx]
    mylist=list(set(mgroup4['End Time PT']))
    for x in mylist:
        ph=mgroup2[mgroup2['End Time PT']==x] 
        L=list((range(len(ph))))
        ph.sort_values(by=['rental_started_at'],inplace=True)
        ph['num']=L
        PH=PH.append(ph)
  
    
"""  
check=DF[DF['vehicle_id']==407]

check.drop_duplicates(inplace=True)


check.drop_duplicates(['rental_id'],keep= 'last',inplace=True)

A=x[['rental_started_at','End Time PT','num']]
"""




features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez','num']
label=['ratings']


X=PH[features]
y=PH['ratings']



from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

#os = SMOTE(random_state=0)


us=RandomUnderSampler(random_state=0)


ux,uy=us.fit_sample(X,y)


columns=features.copy()
columns.extend(label)
FD=PH[columns].head(len(uy))

FD['ratings']=uy
FD[features]=ux


X=FD[features]
y=FD['ratings']


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()

from sklearn.feature_selection import RFE
rfe = RFE(model, 4)
fit = rfe.fit(X, y)
print("Num Features: %d" % fit.n_features_) 
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

import numpy as np
features=np.array(features)
fsupport=fit.support_
selected_features=features[fsupport]



X=FD[selected_features]


#X=FD[features]
y=FD['ratings']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


clf=DecisionTreeClassifier(criterion='entropy')

clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)


"""
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100,criterion='entropy',min_samples_leaf=10,min_samples_split=10)
clf=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=42)
clf=RandomForestClassifier(n_estimators=100,random_state=42)
clf.fit(X_train, y_train)
"""




from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
actual = y_test
predicted = clf.predict(X_test)
results = confusion_matrix(actual, predicted) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(actual, predicted) )
print('Report : ')
print(classification_report(actual, predicted))



"""

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
actual = y_train
predicted = clf.predict(X_train)
results = confusion_matrix(actual, predicted) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(actual, predicted) )
print('Report : ')
print(classification_report(actual, predicted))

"""
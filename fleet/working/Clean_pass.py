#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:50:20 2019

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
df['ez']=df['end_address'].apply(zip)



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



DF['myear']=DF.apply(lambda x:datetime.datetime.strptime(str(x['rental_ended_at'])[0:7], '%Y-%m'),axis=1)




DF['ratings']=DF['rating_score'].apply(lambda x:1 if x>3.0 else 0)

collections.Counter(DF['ratings'])


import pickle

dbfile = open('processed_dataframe', 'rb')      
DF = pickle.load(dbfile) 





"""
Adding an extra feature

"""

dd=pd.read_csv('DarwinJobssmartquery_modified_2019-11-15_0845.csv')

dd.dropna(subset=['closed_at'],inplace=True)
dd['closed_at']=dd['closed_at'].apply(lambda x:datetime.datetime.strptime(x[0:19], '%Y-%m-%d %H:%M:%S'))


DT_COLS=['closed_at']

"""
Converting to American Time from UTC
"""

convert_datetime_columns(dd, DT_COLS)

    

from datetime import datetime

dtt=pd.DataFrame(data=[datetime(2019, 4,1)],columns=['date'])

dtt['date']=dtt['date'].dt.tz_localize('UTC').dt.tz_convert(REGION_TIMEZONE)

DF2=DF[DF['rental_ended_at']>dtt['date'].iloc[0]]
DF2['vehicle_id_v']=DF2['vehicle_id']

group=DF2.groupby(['vehicle_id'])

global df2
df2=dd[['closed_at','vehicle_id']]


mgroup=pd.DataFrame()

def days_from_last_cleaning(x):
    
    #print(type(x))
    #print(x.columns)
    
    global mgroup
    
    x.sort_values(by=['rental_started_at'],inplace=True)
    
    O=df2[df2['vehicle_id']==max(x['vehicle_id_v'])]
    
    O.sort_values(by=['closed_at'],inplace=True)
    
    O['rental_started_at']=O['closed_at']
        
    xyz=pd.merge_asof(x, O, on='rental_started_at')
    mgroup=mgroup.append(xyz)
    
        
    
    



group.apply(days_from_last_cleaning)

mgroup2=mgroup.dropna()





"""
Bad processing code which can be optimized?
"""

PH=pd.DataFrame()    

for xx in list(set(mgroup['vehicle_id_x'])):
    mgroup4=mgroup2[mgroup2['vehicle_id_x']==xx]
    mylist=list(set(mgroup4['closed_at']))
    for x in mylist:
        ph=mgroup2[mgroup2['closed_at']==x] 
        L=list((range(len(ph))))
        ph.sort_values(by=['rental_started_at'],inplace=True)
        ph['num']=L
        PH=PH.append(ph)



import pickle
dbfile = open('extra_processed_dataframe', 'ab') 
      
# source, destination 
pickle.dump(PH, dbfile)                      
dbfile.close() 



import pickle

dbfile = open('extra_processed_dataframe', 'rb')      
PH = pickle.load(dbfile) 

features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez','num']
label=['ratings']


features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez']
label=['ratings']



features=['distance driven','duration_hours','num']


X=PH[features]
y=PH['ratings']

X=DF[features]
y=DF['ratings']


"""
DF=DF[DF['rating_score']!=5.0]
"""

from collections import Counter as c

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



from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, y)
print(model.feature_importances_)




selected_features=['duration_hours','fuel_percent_start','ez','distance driven']

selected_features=['duration_hours','fuel_percent_start','Fuel Used','distance driven']


selected_features=['duration_hours','fuel_percent_start']


X=FD[selected_features]
y=FD['ratings']


"""
X=DF[selected_features]
y=DF['ratings']

"""




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


"""
from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
clf.fit(X_train, y_train)

"""


"""
from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier()
clf.fit(X_train,y_train)
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


SCALERS


""""



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
sxtrain=scaler.transform(X_train)

sxtest=scaler.transform(X_test)



"""

Min Max Scaler

from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()
sxtrain = mm_scaler.fit_transform(X_train)


sxtest = mm_scaler.transform(X_test)


"""


"""

Normalizer

from sklearn.preprocessing import Normalizer
n=Normalizer()
sxtrain = n.fit_transform(X_train)


sxtest = n.transform(X_test)


"""


"""
Robust Scaler

from sklearn.preprocessing import RobustScaler
rb = RobustScaler()
sxtrain=rb.fit_transform(X_train)

sxtest=rb.transform(X_test)

"""


from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
clf=LogisticRegression(penalty='l1',C=1,class_weight='balanced')

clf=LogisticRegression()
clf.fit(sxtrain, y_train)



"""
Naive Bayes



from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()



"""

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
actual = y_test
predicted = clf.predict(sxtest)
results = confusion_matrix(actual, predicted) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(actual, predicted) )
print('Report : ')
print(classification_report(actual, predicted))









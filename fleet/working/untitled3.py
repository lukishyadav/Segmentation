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


#DF['ratings']=DF['rating_score'].apply(lambda x:1 if x>3.0 else 0)

#collections.Counter(DF['ratings'])



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

DF3=DF.copy()

DF=DF[DF['rating_score']!=5.0]





    
"""
Getting the top 20 zip codes
"""    
    
    
import operator
x=dict(collections.Counter(DF['ez']))
sorted_x = sorted(x.items(), key=operator.itemgetter(1),reverse=True)    

sorted_dict=dict(sorted_x[0:19])

sorted_dict.keys() 

top20ez=list(sorted_dict.keys())   


DF=DF[DF['ez'].isin(top20ez)]




features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez']
label=['ratings']
#label='rating_score'

#X=DFtrain[features]
#y=DFtrain[label]





"""
One Hot encoding
"""

from sklearn.preprocessing import OneHotEncoder
XX=DF[['ez']]
ohe=OneHotEncoder()
op=ohe.fit_transform(XX).toarray()

list(ohe.get_feature_names())

for pos,value in enumerate(list(ohe.get_feature_names())):
 DF[value]=op[:,pos]


features.extend(list(ohe.get_feature_names()))


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


X=FD[features]
y=FD['ratings']
"""



from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()

from sklearn.feature_selection import RFE
rfe = RFE(model, 10)
fit = rfe.fit(X, y)
print("Num Features: %d" % fit.n_features_) 
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

import numpy as np
features=np.array(features)
fsupport=fit.support_
selected_features=features[fsupport]


"""
selected_features=['distance driven', 'Fuel Used', 'duration_hours']

selected_features=['fuel_percent_start', 'duration_hours', 'hour']
"""

from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, y)
print(model.feature_importances_)


"""
X=FD[selected_features]
y=FD['ratings']

"""

X=DF[selected_features]


#X=FD[features]
y=DF['ratings']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)


clf=DecisionTreeClassifier(criterion='entropy')

clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)

#clf.fit(X,y)

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


#performance_S(model,X,Y,seed,split)


import my_module
my_module.performance_S(DecisionTreeClassifier(),X,y,0,3)

import my_module
my_module.performance(DecisionTreeClassifier(),X,y,0,3)




"""
Cummulative
"""
MD=list(set(DF['myear']))

for x in MD: 
    DFtrain=DF[DF['myear']<=x]
    DFtest=DF[DF['myear']>x]
    if DFtrain.empty or DFtest.empty:
        pass
    else:        
        clf=DecisionTreeClassifier()
        clf.fit(DFtrain[selected_features],DFtrain['ratings'])
        
        actual = DFtest['ratings']
        predicted = clf.predict(DFtest[selected_features])
        results = confusion_matrix(actual, predicted) 
        # print('Confusion Matrix :')
        #print(results) 
        #print('Accuracy Score :',accuracy_score(actual, predicted) )
        print('Train till:',x)
        print('Report : ')
        print(classification_report(actual, predicted))



"""
Non-Cummulative
"""
MD=list(set(DF['myear']))

for x in MD: 
    DFtrain=DF[DF['myear']==x]
    DFtest=DF[DF['myear']!=x]
    if DFtrain.empty or DFtest.empty:
        pass
    else:        
        clf=DecisionTreeClassifier()
        clf.fit(DFtrain[selected_features],DFtrain['ratings'])
        
        actual = DFtest['ratings']
        predicted = clf.predict(DFtest[selected_features])
        results = confusion_matrix(actual, predicted) 
        # print('Confusion Matrix :')
        #print(results) 
        #print('Accuracy Score :',accuracy_score(actual, predicted) )
        print('Train on:',x)
        print('Report : ')
        print(classification_report(actual, predicted))
        
        
        
    

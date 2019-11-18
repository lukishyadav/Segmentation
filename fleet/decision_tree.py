#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:43:24 2019

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

#df=pd.read_csv('PredictiveMaintenanceFeature_modified_2019-10-25_1815.csv')


df=pd.read_csv('ldata.csv')


df.columns=['tenant_id','start_address','vehicle_id','end_address', 'timestamp', 'rental_started_at',
       'rental_ended_at', 'name', 'rental_id',
       'fuel_percent_start', 'fuel_percent_end',
       'Fuel Used', 'distance driven','rating_reason', 'rating_score']

df['rating_reason'].fillna(0,inplace=True)


df.dropna(inplace=True)

fd=df.copy()
fd.dropna(inplace=True)

VEHICLE_DT_COLS=['rental_started_at','rental_ended_at']



df=df[(df['distance driven']>0) & (df['Fuel Used']>0)]


df['rental_started_og']=df['rental_started_at']
df['rental_started_at']=df['rental_started_at'].apply(lambda x:datetime.datetime.strptime(x[0:19], '%Y-%m-%dT%H:%M:%S'))

df['rental_ended_at']=df['rental_ended_at'].apply(lambda x:datetime.datetime.strptime(x[0:19], '%Y-%m-%dT%H:%M:%S'))

df['rental_duration']=df.apply(lambda x:x['rental_ended_at']-x['rental_started_at'],axis=1)

df['hour']=df['rental_ended_at'].apply(lambda x:x.hour)

df['weekday']=df['rental_ended_at'].apply(lambda x:x.weekday())

df['duration_hours']=df['rental_duration'].apply(lambda x:(x.seconds/3600))


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


"""Including negative ratings w.r.t cleanlinesss only"""

cDF=DF[((DF['rating_reason']=='Cleanliness') & (DF['ratings']==0)) | (DF['ratings']==1)]

clean_df=DF[(DF['rating_reason']=='Cleanliness')]

features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez']
label=['ratings']
#label='rating_score'

X=DF[features]
y=DF[label]



"""
Undersampling"""


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

#FD['rating_score']=df['rating_score'].loc[FD.index]
#C=FD[['distance driven','fuel_percent_start','duration_hours','hour','weekday','ratings','ez']].corr()


pd.crosstab(FD['distance driven'], FD['ratings'], normalize='index')


features=['distance driven','fuel_percent_start', 'duration_hours', 'hour', 'weekday','ez']
features2=['distance driven']

X=FD[features]
y=FD['ratings']



"""
Feature Selection

"""

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

features=list(features)

selected_features=['distance driven','fuel_percent_start','duration_hours']


from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, y)
print(model.feature_importances_)


selected_features=['fuel_percent_start','duration_hours','ez']

selected_features=['Fuel Used','duration_hours','ez']





"""

Training model

"""

X=FD[selected_features]


#X=FD[features]
y=FD[label]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
sxtrain=scaler.transform(X_train)

sxtest=scaler.transform(X_test)



from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial').fit(sxtrain, y_train)



from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier()
clf.fit(sxtrain, y_train)


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion='gini',max_depth=23)
clf=DecisionTreeClassifier(max_depth=23)
clf=DecisionTreeClassifier()

clf.fit(sxtrain, y_train)



from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(sxtrain, y_train)

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=50)
clf.fit(sxtrain, y_train)


clf.score(sxtest,y_test)

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

from numpy import linalg as LA
LA.cond(a)


scaler = StandardScaler()
scaler.fit(XX)
sXX=scaler.transform(XX)




LA.cond(sxtrain)

LA.cond(sXX)


"""
non-standardized data
"""


clf=DecisionTreeClassifier()

clf.fit(X_train, y_train)

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

Cleanliness Ratings only!

"""
X=cDF[features]
y=cDF[label]


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

#FD['rating_score']=df['rating_score'].loc[FD.index]
#C=FD[['distance driven','fuel_percent_start','duration_hours','hour','weekday','ratings','ez']].corr()


pd.crosstab(FD['distance driven'], FD['ratings'], normalize='index')


features=['distance driven','fuel_percent_start','Fuel Used','duration_hours', 'hour', 'weekday','ez']
features2=['distance driven']

X=FD[features]
y=FD['ratings']


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()

from sklearn.feature_selection import RFE
rfe = RFE(model, 3,verbose=2)
fit = rfe.fit(X, y)
print("Num Features: %d" % fit.n_features_) 
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

import numpy as np
features=np.array(features)
fsupport=fit.support_
selected_features=features[fsupport]

features=list(features)

selected_features=['distance driven','fuel_percent_start','duration_hours']


from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier(n_estimators=100)
model.fit(X, y.values)
print(model.feature_importances_)




selected_features=['fuel_percent_start','duration_hours','ez']




X=FD[selected_features]


#X=FD[features]
y=FD[label]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
sxtrain=scaler.transform(X_train)

sxtest=scaler.transform(X_test)



from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial').fit(sxtrain, y_train)



from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier()
clf.fit(sxtrain, y_train)


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion='gini',max_depth=23)
clf=DecisionTreeClassifier(max_depth=23)
clf=DecisionTreeClassifier()

clf.fit(sxtrain, y_train)



from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(sxtrain, y_train)

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=50)
clf.fit(sxtrain, y_train)


clf.score(sxtest,y_test)

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

from numpy import linalg as LA
LA.cond(a)


scaler = StandardScaler()
scaler.fit(XX)
sXX=scaler.transform(XX)




LA.cond(sxtrain)

LA.cond(sXX)



dd=pd.read_csv('DarwinCarJobs_2019-11-5_1504.csv')

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
df2=main_df_filtered[['End Time PT','vehicle_id','job type']]

"""
Creating groupbys
"""

group=DF.groupby(['vehicle_id'])


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

group=DF2.groupby(['vehicle_id'])

DF329.sort_values(by=['vehicle_id'])

import time
s=time.time()

mgroup=pd.DataFrame()

def days_from_last_cleaning(x):
    
    global mgroup
    
    x.sort_values(by=['rental_started_at'],inplace=True)
    
    O=df2[df2['vehicle_id']==max(x['vehicle_id'])]
    
    O.sort_values(by=['End Time PT'],inplace=True)
    
    O['rental_started_at']=O['End Time PT']
        
    xyz=pd.merge_asof(x, O, on='rental_started_at')
    mgroup=mgroup.append(xyz)


group.apply(days_from_last_cleaning)

mgroup2=mgroup.dropna()

print(time.time()-s)

# =============================================================================
# def try1(x):
#     
#     print(max(x['vehicle_id']))
# 
# group.apply(try1)
# =============================================================================

"""

using mgroup2

"""
mgroup2['since_cleaning']=mgroup2.apply(lambda x:(x['rental_started_at']-x['End Time PT']).total_seconds()/3600,axis=1)


features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez','since_cleaning']
label=['ratings']

cDF=mgroup2[((mgroup2['rating_reason']=='Cleanliness') & (mgroup2['ratings']==0)) | (mgroup2['ratings']==1)]

mgroup2.to_csv('data_with_added_features.csv',index=False)
#label='rating_score'

X=mgroup2[features]
y=mgroup2[label]


X=cDF[features]
y=cDF[label]


"""
Undersampling"""


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

#os = SMOTE(random_state=0)


us=RandomUnderSampler(random_state=0)


ux,uy=us.fit_sample(X,y)


columns=features.copy()
columns.extend(label)
FD=mgroup2[columns].head(len(uy))

FD['ratings']=uy
FD[features]=ux

#FD['rating_score']=df['rating_score'].loc[FD.index]
#C=FD[['distance driven','fuel_percent_start','duration_hours','hour','weekday','ratings','ez']].corr()


pd.crosstab(FD['distance driven'], FD['ratings'], normalize='index')



features=['distance driven','fuel_percent_start', 'duration_hours', 'hour', 'weekday','ez','since_cleaning']
features2=['distance driven']

X=FD[features]
y=FD['ratings']



"""
Feature Selection

"""

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()

from sklearn.feature_selection import RFE
rfe = RFE(model, 3)
fit = rfe.fit(X, y)
print("Num Features: %d" % fit.n_features_) 
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

import numpy as np
features=np.array(features)
fsupport=fit.support_
selected_features=features[fsupport]

features=list(features)

selected_features=['distance driven','fuel_percent_start','duration_hours']


from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, y)
print(model.feature_importances_)


selected_features=['fuel_percent_start','duration_hours','ez']



selected_features=['ez', 'fuel_percent_start', 'duration_hours',
       'since_cleaning']


"""

Training model

"""

X=FD[selected_features]


#X=FD[features]
y=FD[label]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
sxtrain=scaler.transform(X_train)

sxtest=scaler.transform(X_test)



from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial').fit(sxtrain, y_train)



from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier()
clf.fit(sxtrain, y_train)


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion='gini',max_depth=23)
clf=DecisionTreeClassifier(max_depth=23)
clf=DecisionTreeClassifier()

clf.fit(sxtrain, y_train)

clf.fit(X_train, y_train)


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(sxtrain, y_train)

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=50)
clf.fit(sxtrain, y_train)


clf.score(sxtest,y_test)

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
Considering only cleanliness related negative ratings improves the result?

Binning numerical variable can result in better results?
"""



"""
Days from last cleaning
"""





MD=dict(collections.Counter(FD['ez']))


import operator
sorted_MD = sorted(MD.items(), key=operator.itemgetter(1),reverse=True)

MD20=dict(sorted_MD[0:20])




ML=list(MD.keys())


FD2=FD[FD['ez'].isin(ML[0:19])]


pd.crosstab(FD2['ez'], FD2['ratings'], normalize='index')





scaler = StandardScaler()
scaler.fit(X)
sX=scaler.transform(X)
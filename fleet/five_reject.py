#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:58:20 2019

@author: lukishyadav
"""

"""

App Crashes

tickets

location mismatch   (cutomer reaches a location where car is not available)


Performance improvement by removing 5 ratings! (Massive improvement?)
93% precision, accuracy, f1 score for 0 and 1 by using selected features: 

['distance driven', 'Fuel Used', 'duration_hours']


Random Forest with n_estimators=100 giving better results

    

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



"""Including negative ratings w.r.t cleanlinesss only"""

cDF=DF[((DF['rating_reason']=='Cleanliness') & (DF['ratings']==0)) | (DF['ratings']==1)]

clean_df=DF[(DF['rating_reason']=='Cleanliness')]

features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez']
label=['ratings']
#label='rating_score'

X=DF[features]
y=DF['ratings']

DF['rental_ended_at']






import datetime
print (datetime.date.today() - datetime.timedelta(6*365/12))   .isoformat()

dtt=pd.DataFrame(data=[datetime(2019, 4,1)],columns=['date'])

#from datetime import datetime
dtt=pd.DataFrame(data=[pd.Timestamp(datetime.date.today() - datetime.timedelta(6*365/12))],columns=['date'])

dtt['date']=dtt['date'].dt.tz_localize('UTC').dt.tz_convert(REGION_TIMEZONE)

DF['month']=DF['rental_ended_at'].apply(lambda x:x.month)

DF['year']=DF['rental_ended_at'].apply(lambda x:x.year)

#DF['myear']=DF.apply(lambda x:str(x['month'])+'-'+str(x['year']),axis=1)

DF['myear']=DF.apply(lambda x:datetime.datetime.strptime(str(x['rental_ended_at'])[0:7], '%Y-%m'),axis=1)





DF['ratings']=DF['rating_score'].apply(lambda x:1 if x>3.0 else 0)

collections.Counter(DF['ratings'])

DF3=DF.copy()

DF=DF[DF['rating_score']!=5.0]


"""


Main Process

"""

#DFtrain=DF[DF['myear']<'5-2019']

#DFtest=DF[DF['myear']>='5-2019']




"""
Undersampling"""



features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez']
label=['ratings']
#label='rating_score'

#X=DFtrain[features]
#y=DFtrain[label]

X=DF[features]
y=DF['ratings']


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
"""

pd.crosstab(FD['distance driven'], FD['ratings'], normalize='index')


features=['distance driven','fuel_percent_start', 'duration_hours', 'hour', 'weekday','ez']
features2=['distance driven']

X=FD[features]
y=FD['ratings']
"""


"""
Feature Selection

"""


#X=FD[features]
#y=FD[label]

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


"""

selected_features=['fuel_percent_start','duration_hours','ez']

selected_features=['Fuel Used','duration_hours','ez']

"""



"""

Training model

"""
#DFtrain=DF[DF['myear']<'5-2019']

#DFtest=DF[DF['myear']>='5-2019']

X=DF[selected_features]


#X=FD[features]
y=DF['ratings']

"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

"""




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
sxtrain=scaler.transform(X_train)

sxtest=scaler.transform(X_test)

#sxtest=scaler.transform(X_test)



"""

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial').fit(sxtrain, y_train)



from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier()
clf.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100,criterion='entropy',min_samples_leaf=10,min_samples_split=10)
clf=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=42)
clf=RandomForestClassifier(n_estimators=100,random_state=42)
clf.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train, y_train)

"""


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion='gini',max_depth=23)
clf=DecisionTreeClassifier(max_depth=23)
clf=DecisionTreeClassifier()

clf.fit(sxtrain,y_train)


"""
clf=DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,y_train)
"""




#clf.score(sxtest,y_test)

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



performance_S(model,X,Y,seed,split)


import my_module
my_module.performance_S(DecisionTreeClassifier(),X,y,42,3)




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



"""
Training Model
"""



"""
trainlist=[str(x)+'-'+y for x in [10,11,12] for y in ['2018']]
trainlist.extend([str(x)+'-'+y for x in [1,2,3,4] for y in ['2019']])


testlist=[str(x)+'-'+y for x in [5,6,7,8,9,10,11] for y in ['2019']]
"""
features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez']
label=['ratings']


DFtrain=DF[DF['myear']<=datetime.datetime(2019,8,1)]

DFtest=DF[DF['myear']>datetime.datetime(2019,8,1)]

X=DFtrain[features]


#X=FD[features]
y=DFtrain['ratings']


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

#os = SMOTE(random_state=0)


us=RandomUnderSampler(random_state=0)


ux,uy=us.fit_sample(X,y)


columns=features.copy()
columns.extend(label)
FD=DFtrain[columns].head(len(uy))

FD['ratings']=uy
FD[features]=ux



X=FD[features]
y=FD['ratings']



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


"""

selected_features=['fuel_percent_start','duration_hours','ez']

selected_features=['Fuel Used','duration_hours','ez']

"""

"""
X=FD[features]
"""

X=DFtrain[selected_features]
y=DFtrain['ratings']


"""
X=DFtrain[selected_features]

y=DFtrain['ratings']

"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
sX=scaler.transform(X)

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion='gini',max_depth=23)
clf=DecisionTreeClassifier(max_depth=23)
clf=DecisionTreeClassifier()

clf.fit(sX,y)


"""
clf=DecisionTreeClassifier()
clf.fit(X,y)
"""



"""
Testing of model
"""

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


mlist=list(set(DFtest['myear']))

#mlist=[str(x)+'-'+y for x in [5,6,7,8,9,10,11] for y in ['2019']]

for x in mlist:
    DF_current=DFtest[DFtest['myear']==x]
    X_test=DF_current[selected_features]
    y_test=DF_current['ratings']
    sxtest=scaler.transform(X_test)
    actual = y_test
    predicted = clf.predict(sxtest)
    results = confusion_matrix(actual, predicted) 
    print('Results for:',str(x))
    print('Confusion Matrix :')
    print(results) 
    print('Accuracy Score :',accuracy_score(actual, predicted) )
    print('Report : ')
    print(classification_report(actual, predicted))
    report=classification_report(actual, predicted,output_dict=True)
    output=pd.DataFrame(report).transpose()
    output.to_csv('output/'+str(x)+'.csv')
    
output_df=DF_current.copy()
output_df['predicted']=list(predicted)
sf=list(selected_features)
sf.extend(['predicted','ratings'])
output_df=output_df[sf]
    

"""
mlist=list(set(DFtest['myear']))


for x in mlist:
    DF_current=DFtest[DFtest['myear']==x]
    X_test=DF_current[selected_features]
    y_test=DF_current['ratings']
    #sxtest=scaler.transform(X_test)
    actual = y_test
    predicted = clf.predict(X_test)
    results = confusion_matrix(actual, predicted) 
    print('Results for:',str(x))
    print('Confusion Matrix :')
    print(results) 
    print('Accuracy Score :',accuracy_score(actual, predicted) )
    print('Report : ')
    print(classification_report(actual, predicted))
    report=classification_report(actual, predicted,output_dict=True)
    output=pd.DataFrame(report).transpose()
    output.to_csv('output/'+str(x)+'.csv')

"""










X_test=FD[selected_features]
y_test=FD['ratings']
sxtest=scaler.transform(X_test)
actual = y_test
predicted = clf.predict(sxtest)
results = confusion_matrix(actual, predicted) 
print('Results for:',str(x))
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(actual, predicted) )
print('Report : ')
print(classification_report(actual, predicted))
report=classification_report(actual, predicted,output_dict=True)
output=pd.DataFrame(report).transpose()
output.to_csv('output/'+str(x)+'.csv')



"""
Model is not under trained!

But is it over trained? 
"""



"""
Testing model trainied on unbalanced data!
"""
X_test=DFtrain[selected_features]
y_test=DFtrain['ratings']
sxtest=scaler.transform(X_test)
actual = y_test
predicted = clf.predict(sxtest)
results = confusion_matrix(actual, predicted) 
print('Results for:',str(x))
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(actual, predicted) )
print('Report : ')
print(classification_report(actual, predicted))
report=classification_report(actual, predicted,output_dict=True)


#output=pd.DataFrame(report).transpose()
#output.to_csv('output/'+str(x)+'.csv')


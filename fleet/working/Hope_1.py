#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:04:29 2019

@author: lukishyadav
"""

import pandas as pd

import numpy as np

from collections import Counter as c


import pickle

dbfile = open('processed_dataframe', 'rb')      
DF = pickle.load(dbfile) 

features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez']
label=['ratings']




X=DF[features]
y=DF['ratings']


"""
Oversampling


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

os = SMOTE(random_state=0,sampling_strategy='minority')


#us=RandomUnderSampler(random_state=0)


ox,oy=os.fit_sample(X,y)


columns=features.copy()
columns.extend(label)
FD=DF[columns].head(len(oy))




od=np.concatenate((ox,oy.reshape(-1,1)),axis=1)

FD=pd.DataFrame(od,columns=columns)



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




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



from sklearn.tree import DecisionTreeClassifier
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
DF=DF[DF['rating_score']!=5.0]
"""


"""
Dirty Stuff

"""
import datetime

DFtrain=DF[DF['myear']<=datetime.datetime(2019,8,1)]

DFtest=DF[DF['myear']>datetime.datetime(2019,8,1)]

X=DFtrain[features]
y=DFtrain['ratings']


""" 

Oversampling


"""

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

os = SMOTE(random_state=0,sampling_strategy='minority')


#us=RandomUnderSampler(random_state=0)


ox,oy=os.fit_sample(X,y)


columns=features.copy()
columns.extend(label)
FD=DF[columns].head(len(oy))




od=np.concatenate((ox,oy.reshape(-1,1)),axis=1)

FD=pd.DataFrame(od,columns=columns)


X=FD[selected_features]
y=FD['ratings']




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



from sklearn.tree import DecisionTreeClassifier
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

Observing principle Components

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




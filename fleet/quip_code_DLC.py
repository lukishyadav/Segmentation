#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:02:00 2019

@author: lukishyadav
"""

"""
quip_code needs to be run through in the same kernel, before running this code!
"""

features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez']
label=['ratings']

DF2=DF.copy()


"""

Binning categorical columns

Recall boost (for 0) by doing this?

"""

bincolumns=['distance driven', 'Fuel Used', 'fuel_percent_start',
       'duration_hours', 'hour']

for colname in bincolumns:
    quantile_list = [0, .25, .5, .75, 1.]
    quantiles = DF2[colname].quantile(quantile_list)
    quantiles
    
    
    quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']
    DF2[colname+'_quantile_range'] = pd.qcut(
                                                DF2[colname], 
                                                q=quantile_list)
    DF2[colname+'_quantile_label'] = pd.qcut(
                                                DF2[colname], 
                                                q=quantile_list,       
                                                labels=quantile_labels)





features=[]
for m in bincolumns:
    features.append(m+'_quantile_label')
    

from sklearn import preprocessing
for n  in features:    
    le = preprocessing.LabelEncoder()
    le.fit(DF2[n])
    DF2[n]=list(le.transform(DF2[n]))
    

DFtrain=DF2[DF2['myear']<=datetime.datetime(2019,8,1)]

DFtest=DF2[DF2['myear']>datetime.datetime(2019,8,1)]

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

X=FD[selected_features]
y=FD['ratings']


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


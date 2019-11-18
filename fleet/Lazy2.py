#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:03:31 2019

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


df=pd.read_csv('fdata.csv')


#df.dropna(inplace=True)

# df.isnull().sum()

df.columns=['tenant_id','start_address','end_address', 'timestamp', 'rental_started_at',
       'rental_ended_at', 'name', 'rental_id',
       'fuel_percent_start', 'fuel_percent_end',
       'Fuel Used', 'distance driven','rating_reason', 'rating_score']

df['rating_reason'].fillna(0,inplace=True)


df.dropna(inplace=True)

fd=df.copy()
fd.dropna(inplace=True)

VEHICLE_DT_COLS=['rental_started_at','rental_ended_at']



df=df[(df['distance driven']>0) & (df['Fuel Used']>0)]

#df['rental_ended_at']
#df['distance driven'].hist()

#df['Fuel Used']

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

"""

import re
for X in range(len(df)):
    s = df['start_address'].iloc[X]
    result = re.search(',([0-9][0-9][0-9][0-9][0-9])', s)
    re.findall(r'[0-9][0-9][0-9][0-9][0-9]', s)[0]
    #print(result.group(1))


re.findall(r'[0-9][0-9][0-9][0-9][0-9]', s)[0]

"""

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

DF=DF[(DF['end_zip']!='missing')]

"""
DF['ez']=DF['end_zip'].apply(lambda x:eval(x))

DF=outlier_trimmean(DF,'ez',5)
"""

# Import label encoder 
from sklearn import preprocessing 

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
DF['ez']= label_encoder.fit_transform(DF['end_zip']) 



#DF['ez']

"""
max(DF['distance driven'])

min(DF['distance driven'])

max(DF['Fuel Used'])

min(DF['Fuel Used'])
"""


#DF['distance driven'].hist(bins=100)

#DF['Fuel Used'].hist(bins=100)



#collections.Counter(DF['rating_score'])


DF['ratings']=DF['rating_score'].apply(lambda x:1 if x>3.0 else 0)

collections.Counter(DF['ratings'])


features=['distance driven','Fuel Used','fuel_percent_start','duration_hours','hour','weekday','ez']
label=['ratings']
#label='rating_score'

X=DF[features]
y=DF[label]


DF2=DF[['distance driven','Fuel Used','duration_hours','hour','weekday','ratings']]


import my_module


import warnings
warnings.filterwarnings("ignore")

"""
Naive Lazy Fit with scaling
"""

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=21)



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



scaler = StandardScaler()
scaler.fit(X)
sx=scaler.transform(X)

my_module.performance_S(LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial'),sx,y,7)





"""
Naive Lazy Fit without scaling
"""


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



from sklearn.linear_model import LogisticRegression
    
clf = LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial').fit(X_train, y_train)


clf.score(X_test,y_test)



"""
Label Balancing and then classification (undersampling)

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

#FD['rating_score']=df['rating_score'].loc[FD.index]
#C=FD[['distance driven','fuel_percent_start','duration_hours','hour','weekday','ratings','ez']].corr()


pd.crosstab(FD['distance driven'], FD['ratings'], normalize='index')


features=['distance driven','fuel_percent_start', 'duration_hours', 'hour', 'weekday','ez']
features2=['distance driven']

X=FD[features]
y=FD['rating']


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



scaler = StandardScaler()
scaler.fit(X_train)
sxtrain=scaler.transform(X_train)

sxtest=scaler.transform(X_test)



from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial').fit(sxtrain, y_train)


clf = LogisticRegression().fit(sxtrain, y_train)



from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion='gini',max_depth=23)
clf=DecisionTreeClassifier(max_depth=23)
clf=DecisionTreeClassifier()

clf.fit(sxtrain, y_train)


from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier()
clf.fit(sxtrain,y_train)

clf.score(sxtest,y_test)

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
actual = y_test
predicted = clf.predict(sxtest)
results = confusion_matrix(actual, predicted) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Scosre :',accuracy_score(actual, predicted) )
print('Report : ')
print(classification_report(actual, predicted))



scaler = StandardScaler()
scaler.fit(X)
sx=scaler.transform(X)

my_module.performance_S(LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial'),sx,y,7)




"""
Label Balancing and then classification (oversampling)

"""

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

os = SMOTE(random_state=0,sampling_strategy='minority')


#us=RandomUnderSampler(random_state=0)


ox,oy=os.fit_sample(X,y)


columns=features.copy()
columns.extend([label])
FD=DF[columns].head(len(oy))




od=np.concatenate((ox,oy.reshape(-1,1)),axis=1)

FD=pd.DataFrame(od,columns=columns)

#FD['ratings']=oy
#FD[features]=ox

FD['rating_score']=df['rating_score'].loc[FD.index]
C=FD[['distance driven','Fuel Used','duration_hours','hour','weekday','ratings','rating_score']].corr()


pd.crosstab(FD['distance driven'], FD['ratings'], normalize='index')


features=['distance driven', 'Fuel Used', 'duration_hours', 'hour', 'weekday']
features2=['distance driven']

X=FD[features2]
y=FD['ratings']


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



scaler = StandardScaler()
scaler.fit(X_train)
sxtrain=scaler.transform(X_train)

sxtest=scaler.transform(X_test)



from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial').fit(sxtrain, y_train)


clf = LogisticRegression().fit(sxtrain, y_train)

from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier()
clf.fit(sxtrain,y_train)

clf.score(sxtest,y_test)

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
actual = y_test
predicted = clf.predict(sxtest)
results = confusion_matrix(actual, predicted) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Scosre :',accuracy_score(actual, predicted) )
print('Report : ')
print(classification_report(actual, predicted))



scaler = StandardScaler()
scaler.fit(X)
sx=scaler.transform(X)

my_module.performance_S(LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial'),sx,y,7)




"""

Using Original Labels

"""



DF['rating_score']=df['rating_score'].copy()


features=['distance driven','Fuel Used','duration_hours','hour','weekday']
label=['rating_score']


X=DF[features]
y=DF[label]



from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

os = SMOTE(random_state=0)


us=RandomUnderSampler(random_state=0)


ux,uy=us.fit_sample(X,y)


columns=features.copy()
columns.extend(label)
FD=DF[columns].head(len(uy))

FD['rating_score']=uy
FD[features]=ux



features=['distance driven', 'Fuel Used', 'duration_hours', 'hour', 'weekday']
features2=['distance driven']

X=FD[features2]
y=FD['rating_score']


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



scaler = StandardScaler()
scaler.fit(X_train)
sxtrain=scaler.transform(X_train)

sxtest=scaler.transform(X_test)



from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial').fit(sxtrain, y_train)


clf = LogisticRegression().fit(sxtrain, y_train)

from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier()
clf.fit(sxtrain,y_train)

clf.score(sxtest,y_test)

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
actual = y_test
predicted = clf.predict(sxtest)
results = confusion_matrix(actual, predicted) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Scosre :',accuracy_score(actual, predicted) )
print('Report : ')
print(classification_report(actual, predicted))



scaler = StandardScaler()
scaler.fit(X)
sx=scaler.transform(X)

my_module.performance_S(LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial'),sx,y,7)




"""


Cross Tabs


"""



from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

os = SMOTE(random_state=0)


us=RandomUnderSampler(random_state=0)


ux,uy=us.fit_sample(X,y)


columns=features.copy()
columns.extend(label)
FD=DF[columns].head(len(uy))

FD['ratings']=uy
FD[features]=ux

FD['rating_score']=df['rating_score'].loc[FD.index]
C=FD[['distance driven','Fuel Used','duration_hours','hour','weekday','ratings','rating_score']].corr()


pd.crosstab(FD['distance driven'], FD['ratings'], normalize='index')





def dd(x):
    if x<=2.5:
        return 1
    elif x>2.5 and x<=7.5:
        return 2
    else:
        return 3


def fu(x):
    if x>=0 and x<1.5:
        return 1
    elif x>1.5 and x<=3:
        return 2
    elif x>3 and x<=4.5:
        return 3
    else:
        return 4


    
FD['b_distance driven']=FD['distance driven'].apply(dd)

FD['b_Fuel Used']=FD['Fuel Used'].apply(fu)


pd.crosstab(FD['b_distance driven'], FD['ratings'], normalize='index')





"""
PCA


"""
X=FD[features]
y=FD['ratings']


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)  

print(pca.explained_variance_ratio_) 

#print(pca.singular_values_)  


px=pca.transform(X)


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(px, y, test_size=0.33, random_state=42)



scaler = StandardScaler()
scaler.fit(X_train)
sxtrain=scaler.transform(X_train)

sxtest=scaler.transform(X_test)



from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial').fit(sxtrain, y_train)


clf = LogisticRegression().fit(sxtrain, y_train)

from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier()
clf.fit(sxtrain,y_train)

clf.score(sxtest,y_test)

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
actual = y_test
predicted = clf.predict(sxtest)
results = confusion_matrix(actual, predicted) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Scosre :',accuracy_score(actual, predicted) )
print('Report : ')
print(classification_report(actual, predicted))



scaler = StandardScaler()
scaler.fit(X)
sx=scaler.transform(X)

my_module.performance_S(LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial'),sx,y,7)
from matplotlib import pyplot as plt

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

plt.plot(FD['distance driven'].values,FD['hour'].values)






"""
Investigations based on discussion.

"""

 #'Fuel Used',
 
 
features=['distance driven','fuel_percent_start','duration_hours','hour','weekday','ez']
label='ratings'

X=FD[features]
y=FD[label]


DF2=DF[['distance driven','Fuel Used','duration_hours','hour','weekday','ratings']]



"""
Feature selection

"""

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier()




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

Balancing the dataset

"""


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

os = SMOTE(random_state=0)


us=RandomUnderSampler(random_state=0)


ux,uy=us.fit_sample(X,y)


columns=features.copy()
columns.extend([label])
FD=DF[columns].head(len(uy))

FD['ratings']=uy
FD[features]=ux

FD['rating_score']=df['rating_score'].loc[FD.index]
C=FD[['distance driven','Fuel Used','duration_hours','hour','weekday','ratings','rating_score']].corr()



selected_features=['distance driven','fuel_percent_start','duration_hours']

selected_features=['fuel_percent_start','duration_hours']

selected_features=['fuel_percent_start','duration_hours','ez']


X=FD[selected_features]


X=FD[features]
y=FD[label]


import my_module


import warnings
warnings.filterwarnings("ignore")


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



"""
Without Scaling

from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier()
clf.fit(X_train,y_train)

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


W*X=Y


scaler = StandardScaler()
scaler.fit(X)
sx=scaler.transform(X)


mod=LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial')
mod=DecisionTreeClassifier()
my_module.performance_S(mod,sx,y,42,2)

#with balacing i.e for non-stratified Kfold, sampling randomness is quiet apparent.
my_module.performance(mod,sx,y,42,5)



"""
Gradient boosting with selected features giving the best result? 

Decision Tree? Giving the best results?

Entropy working better than gini?


['fuel_percent_start', 'duration_hours', 'ez']

decision tree performing better with these features


"""



"""

Deeper into decision Tree

"""


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)



y_pred = dt.predict(X_test)


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


"""

Max Depths


"""




max_depths = np.linspace(1, 64, 64, endpoint=True)


from matplotlib import pyplot as plt

train_results = []
test_results = []
for max_depth in max_depths:
   dt = DecisionTreeClassifier(max_depth=max_depth)
   dt.fit(X_train, y_train)
   train_pred = dt.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous train results
   train_results.append(roc_auc)
   y_pred = dt.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous test results
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1,=plt.plot(max_depths, train_results, 'b', label="Train AUC")
line2,=plt.plot(max_depths, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()




"""
min_samples_split (float)

"""


min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
   dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
   dt.fit(X_train, y_train)
   train_pred = dt.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds =    roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = dt.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.show()



"""
min_samples_split (int)

"""


min_samples_splits = np.linspace(1, 100, 100, endpoint=True)

min_samples_splits=min_samples_splits[1:-1]
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
   min_samples_split=int(min_samples_split) 
   dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
   dt.fit(X_train, y_train)
   train_pred = dt.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds =    roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = dt.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.show()



"""

min_samples_leaf

"""

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
   dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
   dt.fit(X_train, y_train)
   train_pred = dt.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = dt.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples leaf')
plt.show()








""""

sc
""""

MD=dict(collections.Counter(FD['ez']))


import operator
sorted_MD = sorted(MD.items(), key=operator.itemgetter(1),reverse=True)

MD20=dict(sorted_MD[0:20])

ML=list(MD.keys())


FD2=FD[FD['ez'].isin(ML[0:19])]


pd.crosstab(FD2['ez'], FD2['ratings'], normalize='index')
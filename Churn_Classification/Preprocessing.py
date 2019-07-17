#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:05:58 2019

@author: lukishyadav
"""

import pandas as pd
import my_module
import datetime
import seaborn as sns


df=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Churn_Classification/churn_prediction.csv')

"""
Index(['customer_id', 'city', 'signup_date', 'last_trip_date', 'avg_dist',
       'avg_rating_of_driver', 'weekday_pct', 'total_trips',
       'trips_in_first_30_days'],
      dtype='object')

"""


fmt = '%Y-%m-%d %H:%M:%S'

df['signup_date']=df['signup_date'].apply(lambda x:datetime.datetime.strptime(x[0:19], fmt))

df['last_trip_date']=df['last_trip_date'].apply(lambda x:datetime.datetime.strptime(x[0:19], fmt))

from datetime import date
now=datetime.datetime.now()
now=now.strftime('%Y-%m-%d %H:%M:%S')
now=datetime.datetime.strptime(now, fmt)

#then = datetime.datetime.strptime('2018-12-31 23:59:59', fmt)

df['days_from_last_rental']=df['last_trip_date'].apply(lambda x:now-x)

def convert(x):
    return x.total_seconds()/(3600*24)

df['days_from_last_rental']=df['days_from_last_rental'].apply(convert)




DF=df.copy()

DF['Churn']=DF['days_from_last_rental'].apply(lambda x:1 if x>60 else 0)


import collections

collections.Counter(DF['Churn'])


DF.groupby('Churn').describe().T


DF.dropna(subset=['avg_rating_of_driver'],inplace=True)

DF['trips_in_first_30_days'].fillna(0,inplace=True)



m,n,avg_rating_of_driver=my_module.remove_outliers(DF,'avg_rating_of_driver')
m,n,weekday_pct=my_module.remove_outliers(DF,'weekday_pct')
m,n,total_trips=my_module.remove_outliers(DF,'total_trips')
m,n,trips_in_first_30_days=my_module.remove_outliers(DF,'trips_in_first_30_days')


DATA=pd.merge(avg_rating_of_driver[['customer_id','avg_rating_of_driver','Churn']],weekday_pct[['customer_id','weekday_pct']], on='customer_id', how='inner')

DATA=pd.merge(DATA,total_trips[['customer_id','total_trips']], on='customer_id', how='inner')
DATA=pd.merge(DATA,trips_in_first_30_days[['customer_id','trips_in_first_30_days']], on='customer_id', how='inner')



X=DATA[['avg_rating_of_driver','weekday_pct',
       'total_trips', 'trips_in_first_30_days']]
Y=DATA['Churn']

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X, Y)

clf.score(X,Y)
clf = LogisticRegression(random_state=0,C=0.1)
seed=42

from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)


my_module.performance_S(clf,X,Y,seed)

my_module.performance_S(RF,X,Y,seed)

L=my_module.LR(penalty='l1',C=2,max_iter=100,th=0.52) 
my_module.performance_S(L,X,Y,seed)

collections.Counter(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

clf.fit(X_train,Y_train)



from sklearn.metrics import recall_score
Yf=Y_test.iloc[0:100]
Xf=X_test.iloc[0:100,:]


recall_score(Y_test.iloc[0:100],clf.predict(X_test.iloc[0:100,:]))

Xf['True']=Yf
Xf['Predicted']=clf.predict(X_test.iloc[0:100,:])

Xf.reset_index(inplace=True)

len(Xf[Xf['True']==1])

len(Xf[(Xf['True']==1) & (Xf['Predicted']==0)])



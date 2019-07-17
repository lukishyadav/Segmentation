#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:27:11 2019

@author: lukishyadav
"""

import pandas as pd
import my_module
import datetime
import seaborn as sns

DFF=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Customer_Churn_Verification/Excluding19Avergaedaysbetweenrental_modified_2019-7-10_1643.csv')

DFF=DFF[~DFF['Average time between rentals'].str.contains("-")]

DFF['Average time between rentals']=pd.to_timedelta(DFF['Average time between rentals'])
def convert(x):
    return x.total_seconds()/(3600*24)
DFF['a_t_b_r']=DFF['Average time between rentals'].apply(convert)

DFF.drop(columns=['max'],axis=1,inplace=True)

DFF.columns=['customer_id','Average time between rentals','a_t_b_r']



DF=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Customer_Churn_Verification/Excluding19RentalCount_modified_2019-7-10_1633.csv')


df=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Customer_Churn_Verification/Excluding19DayofLastrental_modified_2019-7-10_1635.csv')


fmt = '%Y-%m-%d %H:%M:%S'
then = datetime.datetime.strptime('2018-12-31 23:59:59', fmt)

df['Days from last rental'].iloc[1][0:19]

df['Days from last rental']=df['Days from last rental'].apply(lambda x:datetime.datetime.strptime(x[0:19], fmt))


df['Days from last rental']=df['Days from last rental'].apply(lambda x:then-x)


df['Days from last rental']=pd.to_timedelta(df['Days from last rental'])

def convert(x):
    return x.total_seconds()/(3600*24)

df['d_f_l_r']=df['Days from last rental'].apply(convert)


len(df[df['d_f_l_r']<0])

df=df[df['d_f_l_r']>0]



Data=pd.merge(DFF[['customer_id','a_t_b_r']],df[['customer_id','d_f_l_r']], on='customer_id', how='inner')

Data=pd.merge(Data[['customer_id','a_t_b_r','d_f_l_r']],DF[['customer_id','rental_count']], on='customer_id', how='inner')

m,n,a_t_b_r=my_module.remove_outliers(Data,'a_t_b_r')
m,n,d_f_l_r=my_module.remove_outliers(Data,'d_f_l_r')
m,n,rental_count=my_module.remove_outliers(Data,'rental_count')

DATA=pd.merge(a_t_b_r[['customer_id','a_t_b_r']],d_f_l_r[['customer_id','d_f_l_r']], on='customer_id', how='inner')
DATA=pd.merge(DATA[['customer_id','a_t_b_r','d_f_l_r']],rental_count[['customer_id','rental_count']], on='customer_id', how='inner')


%matplotlib auto
X=DATA[['a_t_b_r', 'd_f_l_r', 'rental_count']].values
from sklearn.preprocessing import MinMaxScaler
M=MinMaxScaler(feature_range=(0, 1), copy=True)
M.fit(X)
xX=M.transform(X)


from sklearn.cluster import AgglomerativeClustering
algo=AgglomerativeClustering(n_clusters=5)
h=my_module.T_Cluster(xX,algo,x_l='average days between rentals',y_l='days from last rental',z_l='rental_count')


List=list(h)

LIST=[index for index, value in enumerate(List) if value == 0]

Check=DATA.ix[LIST]    


Check.to_csv('Cluster2.csv',index=False)







X=DATA[['a_t_b_r', 'd_f_l_r']].values
from sklearn.preprocessing import MinMaxScaler
M=MinMaxScaler(feature_range=(0, 1), copy=True)
M.fit(X)
xX=M.transform(X)

#xX=DATA[['a_t_b_r', 'd_f_l_r']].values


from sklearn.cluster import KMeans
algo=KMeans(n_clusters=4, random_state=0)
h=my_module.Cluster(xX,algo,x_l='average_days_from_Last_rental',y_l='days from last rental')




from sklearn.cluster import AgglomerativeClustering
algo=AgglomerativeClustering(n_clusters=5)
h=my_module.Cluster(xX,algo,x_l='average_days_from_Last_rental',y_l='days from last rental')






Excluding19Avergaedaysbetweenrental_modified_2019-7-10_1643

Excluding19DayofLastrental_modified_2019-7-10_1635

Excluding19RentalCount_modified_2019-7-10_1633


"""
Scatter verification
"""

XXX=Check[['a_t_b_r', 'd_f_l_r']].values
XXXX=M.transform(XXX)
import matplotlib.pyplot as plt
plt.scatter(XXXX[:,0],XXXX[:,1])

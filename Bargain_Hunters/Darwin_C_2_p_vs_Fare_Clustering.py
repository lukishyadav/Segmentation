#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:23:03 2019

@author: lukishyadav
"""

import pandas as pd
import numpy as np
import my_module
# =============================================================================
# df=pd.read_csv('DarwinRentalFare_modified_2019-6-28_1128.csv')
# df=pd.read_csv('DarwinRentalFare_2019-6-28_1144.csv')
# df2=pd.read_csv('DarwinRentalFare_modified_2019-6-28_1200.csv')
# df=pd.read_csv('DARWINRentalwithPromo_modified_2019-6-28_1214.csv')
# =============================================================================
df=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Bargain_Hunters/RentalswithandwithoutPromoCodes_2019-6-28_1238.csv')

# ['Rental ID', 'Rental Started (Pacific Time)', 'Customer ID', 'Fare',
#'Total to Charge', 'Total Credits Used', 'Codes Used']

df['Fare'].isnull().sum()

df.dropna(subset=['Fare'],inplace=True)

df['Total Credits Used'].fillna(0,inplace=True)



df['Coupon_to_pocket']=df[['Fare','Total Credits Used']].apply(lambda x:float(x[1]/x[0]),axis=1)


len(df[df['Total to Charge']<0])

D=df[df['Total to Charge']<0]


D1=df[(df['Fare'].astype(float)-df['Total Credits Used'].astype(float)).round(2)!=df['Total to Charge'].astype(float)]
 
D1.to_csv('/Users/lukishyadav/Desktop/Gittable_Work/Bargain_Hunters/Different_Bhaviour.csv',index=False)
   
D1.dtypes
     


DF=df[(df['Fare'].astype(float)-df['Total Credits Used'].astype(float)).round(2)==df['Total to Charge'].astype(float)]

# =============================================================================
# DF2=DF[DF['Coupon_to_pocket']>1]
# DF2.to_csv('outliers.csv',index=False)
# 
# =============================================================================

DF['flag']=DF['Coupon_to_pocket'].apply(lambda x:1 if x>1 else 0)

DFF=DF[DF['flag']==0]



DFFF=DF[DF['flag']==1]

DFFF.to_csv('/Users/lukishyadav/Desktop/Gittable_Work/Bargain_Hunters/COupon_to_Pocket_high.csv',index=False)
"""

['Rental ID', 'Rental Started (Pacific Time)', 'Customer ID', 'Fare',
       'Total to Charge', 'Total Credits Used', 'Codes Used',
       'Coupon_to_pocket', 'flag']
"""

C=DFF[DFF['Coupon_to_pocket'].isnull()]
C.to_csv('/Users/lukishyadav/Desktop/Gittable_Work/Bargain_Hunters/Zero_fare_Customers.csv')

C2=DFF[DFF['Fare']==0]

#C_NO=DFF.groupby(['Customer ID']).size().reset_index(name='counts')

data=DFF.groupby(['Customer ID'])['Fare','Coupon_to_pocket'].agg('mean').reset_index()

m,n,c2p=my_module.remove_outliers(data,'Coupon_to_pocket') 
m,n,fare=my_module.remove_outliers(data,'Fare') 

Data=pd.merge(c2p[['Customer ID','Coupon_to_pocket']],fare[['Customer ID','Fare']], on='Customer ID', how='inner')

#DFFF=pd.merge(revenue_sum[['customer_id','sum']],coupon_credit[['customer_id','Total Credits Used']], on='customer_id', how='inner')

Data.dropna(inplace=True)

Data.reset_index(inplace=True)

%matplotlib auto
X=Data[['Coupon_to_pocket','Fare']].values
from sklearn.preprocessing import MinMaxScaler
M=MinMaxScaler(feature_range=(0, 1), copy=True)
M.fit(X)
xX=M.transform(X)
 
from sklearn.cluster import KMeans
algo=KMeans(n_clusters=6, random_state=0)
h=my_module.Cluster(xX,algo,x_l='Average Coupon_to_pocket',y_l='Average Fare')



List=list(h)

LIST=[index for index, value in enumerate(List) if value == 1]

Check=Data.ix[LIST]  

Check.to_csv('/Users/lukishyadav/Desktop/Gittable_Work/Bargain_Hunters/Bargain_Hunters.csv',index=False)  

LIST[1]
LIST[0]


M.transform([[0.1,42]])




XXX=Check[['Coupon_to_pocket','Fare']].values
XXXX=M.transform(XXX)
import matplotlib.pyplot as plt
plt.scatter(XXXX[:,0],XXXX[:,1])


#Check.to_csv('Cluster2.csv',index=False)

# =============================================================================
# import collections
# LLL=dict(collections.Counter(h))
# 
# data.isnull().sum()
# =============================================================================



20 people



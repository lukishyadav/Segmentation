#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:16:38 2019

@author: lukishyadav
"""


import pandas as pd
import my_module
import datetime
import seaborn as sns



df=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Customer_Churn_Verification/DayoflastrentalDateFilter_2019-7-9_1231.csv')

fmt = '%Y-%m-%d %H:%M:%S'
then = datetime.datetime.strptime('2018-12-31 23:59:59', fmt)

df['Days from last rental'].iloc[1][0:19]

df['Days from last rental']=df['Days from last rental'].apply(lambda x:datetime.datetime.strptime(x[0:19], fmt))


df['Days from last rental']=df['Days from last rental'].apply(lambda x:then-x)


df['Days from last rental']=pd.to_timedelta(df['Days from last rental'])

def convert(x):
    return x.total_seconds()/(3600*24)

df['d_f_l_r']=df['Days from last rental'].apply(convert)

threshold=60
DF=df[df['d_f_l_r']>threshold]





fd=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Customer_Churn_Verification/daysfromlastrental_modified_2019-6-25_2145.csv')

fd['Days from last rental ']=pd.to_timedelta(fd['Days from last rental '])

def convert(x):
    return x.total_seconds()/(3600*24)

fd['d_f_l_r']=fd['Days from last rental '].apply(convert)

FD=fd[fd['d_f_l_r']>threshold+175]

Data=pd.merge(DF[['customer_id']],FD[['customer_id','d_f_l_r']], on='customer_id', how='inner')

FD2=fd.copy()


Data2=pd.merge(DF[['customer_id','d_f_l_r']],FD2[['customer_id','d_f_l_r']], on='customer_id', how='inner')

Data2['Factor']=Data2[['d_f_l_r_x', 'd_f_l_r_y']].apply(lambda x:x[1]-x[0],axis=1)

from scipy import stats
#mode=float(stats.mode(Data2['Factor']))

import statistics
mode=statistics.mode(round(Data2['Factor'],3))


Data2['Factor']=Data2['Factor'].apply(lambda x:round(x,3))

r1=len(Data2[Data2['Factor']==mode])

r2=len(Data2['Factor'])


Ratio_Threshold=r1/r2

print(Ratio_Threshold)




dff=pd.read_csv('DarwinRentalCount_modified_2019-7-3_1409.csv')

Data3=pd.merge(Data2[['customer_id', 'd_f_l_r_x', 'd_f_l_r_y', 'Factor']],dff[['customer_id','rental_count']], on='customer_id', how='inner')



rental_count=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Customer_Churn_Verification/RentalCountwithFilter_2019-7-9_1530.csv')

rental_count.columns=['customer_id', 'rental_count_2019', 'min']

Data4=pd.merge(Data3[['customer_id', 'd_f_l_r_x', 'd_f_l_r_y', 'Factor', 'rental_count']],rental_count[['customer_id','rental_count_2019']], on='customer_id', how='left')

Data4.fillna(0,inplace=True)

r1=len(Data4[(Data4['Factor']==mode) & (Data4['rental_count_2019']==float(0))])
Ratio_Threshold=r1/r2

print(Ratio_Threshold)

#100 percent churn


Data3.columns


Check=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Customer_Churn_Verification/Cluster2.csv')


Data5=pd.merge(Check[['customer_id','d_f_l_r','a_t_b_r']],rental_count[['customer_id','rental_count_2019']], on='customer_id', how='inner')





# 75 :  0.8444744775964237

# 90 : 0.8578345070422535

# 94 :  0.8637586902893025

#  96:   0.8651850180505415

#  98:   0.8666439677675633

# 100:    0.8687099725526075

# 102:  0.8702977152088622


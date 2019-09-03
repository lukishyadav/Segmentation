#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:31:12 2019

@author: lukishyadav
"""

#import my_module
import pandas as pd



from bokeh.io import curdoc
import logging
from bokeh.layouts import column,layout,row,widgetbox
import pandas as pd
#import my_module
import datetime
import seaborn as sns
from pyproj import Proj
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.tile_providers import CARTODBPOSITRON ,CARTODBPOSITRON_RETINA
import numpy as np
from sklearn.cluster import DBSCAN 
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider,TextInput
from bokeh.palettes import Category20
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from datetime import datetime

df=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/new_churn/TotalRevenueDarwin_2019-8-2_1514.csv')

df=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/new_churn/TotalRevenueDarwin_2019-8-23_1242.csv')



DD=df.describe()

"""
['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
"""


q25=DD.loc['25%','sum']
q50=DD.loc['50%','sum']
q75=DD.loc['75%','sum']


def categorize(x):
    if x<q25:
        return 'q1'
    elif x>q25 and x <q50:
        return 'q2'
    elif x>q50 and x<q75:
        return 'q3'
    else:    
        return 'q4'
    
df['segment']=df['sum'].apply(categorize) 


#df=df.head(72)



a=df[df['segment']=='q1']   
b=df[df['segment']=='q2']   
c=df[df['segment']=='q3']   
d=df[df['segment']=='q4'] 

"""
A=a.groupby('Country')['sum'].agg('sum')
B=b.groupby('Country')['sum'].agg('sum')
C=c.groupby('Country')['sum'].agg('sum')
D=d.groupby('Country')['sum'].agg('sum')
"""

FD=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Bargain_Hunters/RentalswithandwithoutPromoCodes_2019-6-28_1238.csv')

FD=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/new_churn/RentalswithandwithoutPromoCodes_2019-8-23_1245.csv')




"""
'Rental ID', 'Rental Started (Pacific Time)', 'Customer ID', 'Fare',
       'Total to Charge', 'Total Credits Used', 'Codes Used']
"""


#FD['Rental Started (Pacific Time)'].iloc[1][0:10]

fmt = '%Y-%m-%d %H:%M:%S'

FMT = '%Y-%m-%d %H'

FD['start time']=FD['Rental Started (Pacific Time)'].apply(lambda x:datetime.strptime(x[0:13], FMT))

FD['day']=FD['Rental Started (Pacific Time)'].apply(lambda x:datetime.strptime(x[0:10], '%Y-%m-%d'))


FD['Month']=FD['start time'].apply(lambda x:x.month)


FD['Year']=FD['start time'].apply(lambda x:x.year)




FD=FD[FD['Fare']>0]

FD.columns=['Rental ID', 'Rental Started (Pacific Time)', 'customer_id', 'Fare',
       'Total to Charge', 'Total Credits Used', 'Codes Used', 'start time',
       'day', 'Month', 'Year']

#FD.columns

A=pd.merge(a[['customer_id']],FD,on='customer_id',how='inner')  

B=pd.merge(b[['customer_id']],FD,on='customer_id',how='inner')  

C=pd.merge(c[['customer_id']],FD,on='customer_id',how='inner')  

D=pd.merge(d[['customer_id']],FD,on='customer_id',how='inner')   

"""
AA=A.groupby('day')['Fare'].agg('sum').reset_index(name='sum')
BB=B.groupby('day')['Fare'].agg('sum').reset_index(name='sum')
CC=C.groupby('day')['Fare'].agg('sum').reset_index(name='sum')
DD=D.groupby('day')['Fare'].agg('sum').reset_index(name='sum')
"""



AA=A.groupby(['Month','Year'])['Fare'].agg('sum').reset_index(name='sum')
BB=B.groupby(['Month','Year'])['Fare'].agg('sum').reset_index(name='sum')
CC=C.groupby(['Month','Year'])['Fare'].agg('sum').reset_index(name='sum')
DD=D.groupby(['Month','Year'])['Fare'].agg('sum').reset_index(name='sum')



#FDD=FD.groupby('start time')['Total Credits Used'].agg('sum')

import matplotlib.pyplot as plt
%matplotlib auto
plt.plot(AA['sum'],label='q1')
plt.plot(BB['sum'],label='q2')
plt.plot(CC['sum'],label='q3')
plt.plot(DD['sum'],label='q4')
plt.legend()
plt.show()  


dlr=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/new_churn/Daysfromlastrental_modified_2019-8-13_1806.csv')

dlr=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/new_churn/data/Churndayoflastrentalm19_modified_2019-8-22_1744.csv')


fmt = '%Y-%m-%d %H:%M:%S'

then = datetime.strptime('2019-08-22 12:14:00', fmt)


import datetime as DT
today = DT.datetime.today()
strtoday=str(today)


then = datetime.strptime(strtoday[0:19], fmt)


dlr.columns


dlr['Day of last rental '].iloc[1][0:19]


dlr.dropna(inplace=True)

dlr['Day of last rental ']=dlr['Day of last rental '].apply(lambda x:datetime.strptime(x[0:19], fmt))


dlr['Day of last rental ']=dlr['Day of last rental '].apply(lambda x:then-x)


dlr['Day of last rental ']=pd.to_timedelta(dlr['Day of last rental '])

def convert(x):
    return x.total_seconds()/(3600*24)

dlr['d_f_l_r']=dlr['Day of last rental '].apply(convert)


#df


churn=pd.merge(df,dlr,on='customer_id',how='inner')

churn['churned']=churn['d_f_l_r'].apply(lambda x:1 if x>60 else 0)


churn_group=churn.groupby('segment')['d_f_l_r'].agg('mean').reset_index(name='Average')

churn_group=churn.groupby('segment')['churned'].agg('sum').reset_index(name='Churned')



dfr=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/new_churn/data/Daysfromregistration_2019-8-22_1739.csv')

dfr.columns=['customer_id','dfr']

dfr['dfr']=pd.to_timedelta(dfr['dfr'])

def convert(x):
    return x.total_seconds()/(3600*24)

dfr['d_f_r']=dfr['dfr'].apply(convert)




MMain=pd.merge(dlr,dfr,on='customer_id',how='inner')

MMain['Churn']=MMain['d_f_l_r'].apply(lambda x:1 if x>=60 else 0)

MMain2=MMain[MMain['Churn']==1]

MMain2['churn_time']=MMain2[['d_f_l_r','d_f_r']].apply(lambda x:x[1]-x[0],axis=1)

MMain2['churn_time']=MMain2['churn_time'].apply(lambda x:0 if x<0 else x)




MMain3=pd.merge(MMain2,df,on='customer_id',how='inner')


MMain4=MMain3.groupby(['segment'])['churn_time'].agg('mean').reset_index(name='mean')


MMain4.to_csv('/Users/lukishyadav/Desktop/segmentation/new_churn/data/churn_time.csv',index=False)




#Rentals=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/new_churn/data/DarwinRentals_modified_2019-8-26_2242.csv')

Rentals=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/new_churn/data/dayoffirstrental_modified_2019-8-27_1739.csv')
fmt = '%Y-%m-%d %H:%M:%S'
then = datetime.strptime('2019-08-22 12:14:00', fmt)
#import datetime as DT
#today = DT.datetime.today()
#strtoday=str(today)
#then = datetime.strptime(strtoday[0:19], fmt)

Rentals['Days of first rental'].iloc[1][0:19]
Rentals.dropna(inplace=True)
Rentals['Days of first rental']=Rentals['Days of first rental'].apply(lambda x:datetime.strptime(x[0:19], fmt))
Rentals['Days of first rental']=Rentals['Days of first rental'].apply(lambda x:then-x)
Rentals['Days of first rental']=pd.to_timedelta(Rentals['Days of first rental'])
def convert(x):
    return x.total_seconds()/(3600*24)
Rentals['d_f_f_r']=Rentals['Days of first rental'].apply(convert)

Rentals['Days from last rental']=Rentals['Days from last rental'].apply(lambda x:datetime.strptime(x[0:19], fmt))
Rentals['Days from last rental']=Rentals['Days from last rental'].apply(lambda x:then-x)
Rentals['Days from last rental']=pd.to_timedelta(Rentals['Days from last rental'])
Rentals['d_f_l_r']=Rentals['Days from last rental'].apply(convert)



CT=pd.merge(Rentals[['customer_id','d_f_f_r','d_f_l_r']],df,on='customer_id',how='inner')

CT['Churn']=CT['d_f_l_r'].apply(lambda x:1 if x>=60 else 0)

CT2=CT[CT['Churn']==1]


CT2['churn_time']=CT2[['d_f_l_r','d_f_f_r']].apply(lambda x:x[1]-x[0],axis=1)

CT2['churn_time']=CT2['churn_time'].apply(lambda x:0 if x<0 else x)


CT3=pd.merge(CT2,df,on='customer_id',how='inner')

CT4=CT3.groupby(['segment_x'])['churn_time'].agg('mean').reset_index(name='mean')


CT4.to_csv('/Users/lukishyadav/Desktop/segmentation/new_churn/data/churn_time2.csv',index=False)


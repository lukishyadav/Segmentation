#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:31:12 2019

@author: lukishyadav
"""

import my_module
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


A=a.groupby('Country')['sum'].agg('sum')
B=b.groupby('Country')['sum'].agg('sum')
C=c.groupby('Country')['sum'].agg('sum')
D=d.groupby('Country')['sum'].agg('sum')


FD=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Bargain_Hunters/RentalswithandwithoutPromoCodes_2019-6-28_1238.csv')

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


AA=A.groupby('day')['Fare'].agg('sum').reset_index(name='sum')
BB=B.groupby('day')['Fare'].agg('sum').reset_index(name='sum')
CC=C.groupby('day')['Fare'].agg('sum').reset_index(name='sum')
DD=D.groupby('day')['Fare'].agg('sum').reset_index(name='sum')




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



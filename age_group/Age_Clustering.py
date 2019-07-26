#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:23:29 2019

@author: lukishyadav
"""

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
import my_module

df=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/age_group/Darwinnoofrentalsandage_2019-7-24_1519.csv')

df.columns=['customer_id','age','rental_count']



from datetime import date, timedelta

x='1991-05-25'

f='%Y-%m-%d'

def conversion(x):
 birth_date=datetime.datetime.strptime(x, f)
 age = (date.today() - birth_date.date()) // timedelta(days=365.2425)
 return age

df['age']=df['age'].apply(conversion)


m,n,age=my_module.remove_outliers(df,'age') 

m,n,counts=my_module.remove_outliers(df,'rental_count') 


#M,N,Factor=my_module.Remove_Outliers(Commuters,'factor',1) 

sns.boxplot(x=Factor['factor'])


finale=pd.merge(df,counts, on='customer_id', how='inner')




m,n,counts_x=my_module.remove_outliers(Commuters,'counts_x') 

finale=pd.merge(Commuters,counts_x, on='customer_id', how='inner')

finale.reset_index(inplace=True)






age = (date.today() - birth_date) // timedelta(days=365.2425)


%matplotlib auto
X=finale[['age_x','rental_count_x']].values
from sklearn.preprocessing import MinMaxScaler
M=MinMaxScaler(feature_range=(0, 1), copy=True)
M.fit(X)
xX=M.transform(X)

xX=finale[['age_x','rental_count_x']].values
#xX=X`

df['age'].iloc[1]
 
from sklearn.cluster import KMeans
algo=KMeans(n_clusters=2, random_state=0)
h=my_module.Cluster(xX,algo,x_l='Age',y_l='Rental_Counts')


from sklearn.cluster import AgglomerativeClustering
algo = AgglomerativeClustering(n_clusters=4).fit(X)
h=my_module.Cluster(xX,algo,x_l='Age',y_l='Rental_Counts')


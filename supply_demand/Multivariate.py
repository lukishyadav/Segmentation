#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:07:45 2019

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

df=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/supply_demand/supply_demand_counts_20190501_20190606_hex_(-634,-304).csv')



bd=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/supply_demand/last7days.csv')

bd['Date']=bd['time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M'))


DF=df.copy()


from datetime import datetime

DF['Day']=DF['Date'].apply(lambda x:datetime.strptime(x[0:10],'%Y-%m-%d'))

DF['Date']=DF['Date'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))


DF['month']=DF['Date'].apply(lambda x:x.month)
#DF['Weekday']=DF['Date'].apply(lambda x:x.weekday())
#DF['Weekday']=DF['Weekday'].astype(str)

DF['Weekday']=DF['Day'].apply(lambda x:x.strftime("%A"))

Main=pd.merge(DF,bd,on='Date', how='inner')

LL=list(set(DF['Weekday']))

DDD=DF.groupby(['month'])['Demand_Count'].sum()
DDD=DDD.to_frame()
DDD.reset_index(inplace=True)
#DD=DF.groupby(['Day']).size().reset_index(name='counts')


#['Unnamed: 0', 'Date', 'Appopen_Count', 'Supply_Count', 'Rental_Count','hour']

DD=DF.groupby(['Day']).sum()

DD=DF.groupby(['Day'])['Demand_Count'].sum()
#DD.reset_index(inplace=True)
DD=DD.to_frame()
DD.reset_index(inplace=True)
#D['Weekday']=DD['Day'].apply(lambda x:x.weekday())
DD['Weekday']=DD['Day'].apply(lambda x:x.strftime("%A"))




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:04:00 2019

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


df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/Darwin_Demand.csv')


from datetime import datetime

df['Day']=df['date'].apply(lambda x:datetime.strptime(x[0:19],'%Y-%m-%d %H:%M:%S'))


#DF['Date']=DF['Date'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))



fd=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/supply_demand/last7days.csv')

fd['Day']=fd['time'].apply(lambda x:datetime.strptime(x[0:19],'%Y-%m-%d %H:%M'))


DF=pd.merge(df,fd,on='Day',how='inner')

"""

'date', 'counts', 'Day', 'chance_of_rain', 'chance_of_snow', 'cloud',
       'condition', 'dewpoint_c', 'dewpoint_f', 'feelslike_c', 'feelslike_f',
       'gust_kph', 'gust_mph', 'heatindex_c', 'heatindex_f', 'humidity',
       'is_day', 'precip_in', 'precip_mm', 'pressure_in', 'pressure_mb',
       'temp_c', 'temp_f', 'time', 'time_epoch', 'vis_km', 'vis_miles',
       'will_it_rain', 'will_it_snow', 'wind_degree', 'wind_dir', 'wind_kph',
       'wind_mph', 'windchill_c', 'windchill_f']

"""


DD=DF.corr(method ='pearson')


DF.isnull().sum()


DF['counts_s'] = DF['counts'].shift(-1)

FD=DF[['counts','counts_s']]

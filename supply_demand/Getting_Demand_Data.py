#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:33:13 2019

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




#df=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/new_churn/TotalRevenueDarwin_2019-8-2_1514.csv')


#df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/Commuters/commuters.csv')

df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/DarwinRentals_modified_2019-8-5_1347.csv')

df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/data/DarwinRentals_2019-8-19_1506.csv')

df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/data/DarwinRentals_2019-8-20_1022.csv')

df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/data/DarwinRentals_modified_2019-9-12_1332.csv')





df.isnull().sum()

df.dropna(inplace=True)


"""
['customer_id', 'rental_id', 'start location lat/lng',
       'end location lat/lng', 'Time taken from start to End', 'start time',
       'end datetime', 'Distance driven']

"""
#C=df[df['Distance driven']==0]

# settings
regions_info = {
     'oakland': dict(
         x_min=-13618976.4221,
         x_max=-13605638.1607,
         y_min=4549035.0828,
         y_max=4564284.2700,
         timezone='America/Los_Angeles'),
     'madrid': dict(
         x_min=-416448.0394,
         x_max=-406912.5201,
         y_min=4921025.4356,
         y_max=4931545.0816,
         timezone='Europe/Madrid')
}
REGION = 'oakland'
selected_region = regions_info['oakland']


HEX_SIZE = 10000
Q = -634
R = -304


def convert_to_mercator(lngs, lats):
    # converts incoming iterable degrees to mercator
    from pyproj import Proj  # put here for clarity
    
    projection = Proj(init='epsg:3857')
    xs = []
    ys = []
    for lng, lat in zip(lngs, lats):
        x, y = projection(lng, lat)
        xs.append(x)
        ys.append(y)
    return xs, ys


def convert_to_mercator(lngs, lats):
    projection = Proj(init='epsg:3857')
    xs = []
    ys = []
    for lng, lat in zip(lngs, lats):
        x, y = projection(lng, lat)
        xs.append(x)
        ys.append(y)
    return xs, ys


df['s_try']=df['start location lat/lng'].apply(lambda x:x.split(","))

df['s_lat_col']=df['s_try'].apply(lambda x:float(x[0]))

df['s_lng_col']=df['s_try'].apply(lambda x:float(x[1]))


df['s_merc_lng'],df['s_merc_lat']=convert_to_mercator(df['s_lng_col'],df['s_lat_col'])



HEX_SIZE=10000

from bokeh.util.hex import cartesian_to_axial
df['q'], df['r'] = cartesian_to_axial(
    df['s_merc_lng'],
    df['s_merc_lat'],
    size=HEX_SIZE,
    orientation='pointytop'
)


df['start time'].iloc[1][0:13]


fmt = '%Y-%m-%d %H:%M:%S'

FMT='%Y-%m-%d %H'


from datetime import datetime

df['date']=df['start time'].apply(lambda x:datetime.strptime(x[0:13], FMT))

df['start time']=df['start time'].apply(lambda x:datetime.strptime(x[0:19], fmt))



DF = df[(df['q'] == Q) & (df['r'] == R)]


DD=DF.groupby('date').size().reset_index(name='counts')


DD.to_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand/data/Darwin_Demand_19_8_done.csv',index=False)


#DF['hour']=DF['Date'].apply(lambda x:x.hour)



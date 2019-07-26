#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:00:07 2019

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
import dask.dataframe as dd




s=time.time()
df = dd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Commuters/commuters.csv')
print(time.time()-s) 


def convert_to_mercator(lngs, lats):
    projection = Proj(init='epsg:3857')
    xs = []
    ys = []
    for lng, lat in zip(lngs, lats):
        x, y = projection(lng, lat)
        xs.append(x)
        ys.append(y)
    return xs, ys




result = df.x.map_partitions(f)


df['start location lat/lng'].map_partitions(lambda x:x.split(","), meta=pd.Series([], dtype=list, name='ss')).compute()


df['s_try']=df['start location lat/lng'].map(lambda x:x.split(","))


# =============================================================================
# import multiprocessing
# 
# XX=dd.from_pandas(df,npartitions=4*multiprocessing.cpu_count()).map_partitions(lambda df:df.apply((lambda row:row.s_try.split(",")),axis=1))
# 
# 
# XX=df.map_partitions(lambda df:df.apply((lambda row:row.s_try.split(",")),axis=1))
# 
# =============================================================================


#df['s_try']=df['start location lat/lng'].map_partitions(lambda x:x.split(","))

df['s_lat_col']=df['s_try'].map(lambda x:float(x[0]),meta=pd.Series([], dtype=float, name='ss'))

df['s_lng_col']=df['s_try'].map(lambda x:float(x[1]))


df['s_merc_lng'],df['s_merc_lat']=convert_to_mercator(df['s_lng_col'],df['s_lat_col'])



df['e_try']=df['end location lat/lng'].apply(lambda x:x.split(","))

df['e_lat_col']=df['e_try'].apply(lambda x:float(x[0]))

df['e_lng_col']=df['e_try'].apply(lambda x:float(x[1]))


df['e_merc_lng'],df['e_merc_lat']=convert_to_mercator(df['e_lng_col'],df['e_lat_col'])


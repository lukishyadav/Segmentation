#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 17:47:52 2019

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

df=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Commuters/commuters.csv')


"""

['customer_id', 'rental_id', 'start location lat/lng',
       'end location lat/lng', 'Time taken from start to End', 'start time',
       'end datetime', 'Distance driven', 's_try', 's_lat_col', 's_lng_col',
       's_merc_lng', 's_merc_lat', 'e_try', 'e_lat_col', 'e_lng_col',
       'e_merc_lng', 'e_merc_lat']

"""

DD=df.groupby(['customer_id']).size().reset_index(name='counts')

CS=list(set(DD['customer_id'][DD['counts']>4]))

global CUSTOMER
CUSTOMER=33569

#95 gives decent results

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



df['e_try']=df['end location lat/lng'].apply(lambda x:x.split(","))

df['e_lat_col']=df['e_try'].apply(lambda x:float(x[0]))

df['e_lng_col']=df['e_try'].apply(lambda x:float(x[1]))


df['e_merc_lng'],df['e_merc_lat']=convert_to_mercator(df['e_lng_col'],df['e_lat_col'])


maxlat=max(df['s_merc_lat'])
minlat=min(df['s_merc_lat'])

maxlng=max(df['s_merc_lng'])
minlng=min(df['s_merc_lng'])






kms_per_radian = 6371.0088


global min_meters 

def set_clusters(df, percent, min_n, min_meters):
    # set the data up for clustering
    min_percent = int(df.shape[0]/100) * percent  # 1% of samples available
    epsilon = float(min_meters)/1000/kms_per_radian  # m * (1km/1000m) * (radian/km) => radians
    X = np.column_stack((df['s_lng_col'], df['s_lat_col']))
    e_X = np.column_stack((df['e_lng_col'], df['e_lat_col']))
    # clustering
    dbscan = DBSCAN(eps=epsilon, min_samples=max(min_percent, min_n),
                    algorithm='ball_tree',
                    metric='haversine').fit(np.radians(X))
    
    e_dbscan = DBSCAN(eps=epsilon, min_samples=max(min_percent, min_n),
                    algorithm='ball_tree',
                    metric='haversine').fit(np.radians(e_X))

    # create list of clusters and labels
    #n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    #n_noise_ = list(dbscan.labels_).count(-1)

    # label the points unique to their cluster
    #unique_labels = set(dbscan.labels_)
    df['label'] = [str(label) for label in dbscan.labels_]
    df['e_label'] = [str(label) for label in e_dbscan.labels_]

    return df







def my_slider_handler(attr,old,new):
    #slider.value=new
    min_meters=slider.value
    min_n=Min_n.value
    percent=Percent.value
    
    #   source.change.emit()
    filtered_df=set_clusters(df[df['customer_id']==CUSTOMER],percent,min_n,min_meters)
    noise_df = filtered_df[filtered_df['label'] == '-1']
    datapoints_df = filtered_df[filtered_df['label'] != '-1']
    
    e_noise_df = filtered_df[filtered_df['e_label'] == '-1']
    e_datapoints_df = filtered_df[filtered_df['e_label'] != '-1']
    #noise_source = ColumnDataSource()
    #datapoints_source = ColumnDataSource()
    
    noise_source.data = dict(
        x=noise_df['s_merc_lng'],
        y=noise_df['s_merc_lat'],
        label=noise_df['label'],
        #time=noise_df['start_datetime']
        )
    datapoints_source.data = dict(
        x=datapoints_df['s_merc_lng'],
        y=datapoints_df['s_merc_lat'],
        label=datapoints_df['label'],
        #time=datapoints_df['start_datetime']
        )
    
    
    e_noise_source.data = dict(
        x=e_noise_df['e_merc_lng'],
        y=e_noise_df['e_merc_lat'],
        label=e_noise_df['e_label'],
        #time=noise_df['start_datetime']
        )
    e_datapoints_source.data = dict(
        x=e_datapoints_df['e_merc_lng'],
        y=e_datapoints_df['e_merc_lat'],
        label=e_datapoints_df['e_label'],
        #time=datapoints_df['start_datetime']
        )


# create some widgets
slider = Slider(start=50, end=10000, value=50, step=50, title="Epsilon")
slider.on_change("value", my_slider_handler)

Min_n = Slider(start=1, end=1000, value=5, step=1, title="Min_n")
Min_n.on_change("value", my_slider_handler)

Percent = Slider(start=1, end=100, value=5, step=1, title="Percent")
Percent.on_change("value", my_slider_handler)

min_meters=50

filtered_df=set_clusters(df[df['customer_id']==CUSTOMER],5,5,min_meters)

check=df[df['customer_id']==21]

# separate out the noise and the clustered points
noise_df = filtered_df[filtered_df['label'] == '-1']
datapoints_df = filtered_df[filtered_df['label'] != '-1']

e_noise_df = filtered_df[filtered_df['e_label'] == '-1']
e_datapoints_df = filtered_df[filtered_df['e_label'] != '-1']
    

noise_source = ColumnDataSource()
datapoints_source = ColumnDataSource()


e_noise_source = ColumnDataSource()
e_datapoints_source = ColumnDataSource()

# modify columndatasources to modify the figures
noise_source.data = dict(
    x=noise_df['s_merc_lng'],
    y=noise_df['s_merc_lat'],
    label=noise_df['label'],
    #time=noise_df['start_datetime']
    )
datapoints_source.data = dict(
    x=datapoints_df['s_merc_lng'],
    y=datapoints_df['s_merc_lat'],
    label=datapoints_df['label'],
    #time=datapoints_df['start_datetime']
    )


e_noise_source.data = dict(
    x=e_noise_df['e_merc_lng'],
    y=e_noise_df['e_merc_lat'],
    label=e_noise_df['e_label'],
    #time=noise_df['start_datetime']
    )
e_datapoints_source.data = dict(
    x=e_datapoints_df['e_merc_lng'],
    y=e_datapoints_df['e_merc_lat'],
    label=e_datapoints_df['e_label'],
    #time=datapoints_df['start_datetime']
    )



palette_range = [str(x) for x in range(0, 20)]


map_repr='mercator'


# set up/draw the map
map_figure = figure(
    x_range=(minlng,maxlng),
    y_range=(minlat, maxlat),
    x_axis_type=map_repr,
    y_axis_type=map_repr,
    title='Clustering Map Representation'
)
map_figure.add_tile(CARTODBPOSITRON_RETINA)

#show(map_figure)

  
def plot_points(map_figure, noise_source, datapoints_source,e_noise_source,e_datapoints_source):
    noise_point_size = 1
    cluster_point_size = 10

    # plot points on map
    map_figure.circle(x='x', y='y', size=noise_point_size,
                      fill_alpha=0.2, source=noise_source)

    map_figure.circle(x='x', y='y', size=cluster_point_size,
                      fill_alpha=1, source=datapoints_source,
                      fill_color=factor_cmap(
                          'label',
                          palette=Category20[20],
                          factors=palette_range),line_color='black',legend='label')
    

    map_figure.square(x='x', y='y', size=noise_point_size,
                      fill_alpha=0.2, source=e_noise_source)

    map_figure.square(x='x', y='y', size=cluster_point_size,
                      fill_alpha=1, source=e_datapoints_source,
                      fill_color=factor_cmap(
                          'label',
                          palette=Category20[20],
                          factors=palette_range),line_color='red',legend='label')
               
    #show(map_figure)                  
                      
 

                     
plot_points(map_figure, noise_source, datapoints_source,e_noise_source,e_datapoints_source)

#inputs=column(row(widgetbox(slider, width=600)),row(map_figure),width=1000)

layout = row(
            column(
                widgetbox(slider,width=350),
                widgetbox(Min_n, width=300),
                Percent,
                width=400),    
            column(map_figure),
        )

curdoc().add_root(layout)
curdoc().title = 'DOW and Hour Clustering Analysis'

#logging.info('initial map drawn')



"""
minPts = 2·dim can be used
minPts=4 here... (minimum)

"""
#runfile('C:/yourfolder/myfile.py',args='one two three')



import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import itertools

sliced = list(itertools.islice(CS, 100))

#global DF

DF=df[df['customer_id'].isin(sliced)]
DF['label']=np.zeros(len(DF))
DF['e_label']=np.zeros(len(DF))

import time
s=time.time()

for x in sliced:
    #CUSTOMER=x
    
    DF.loc[DF['customer_id']==x]=set_clusters(DF[DF['customer_id']==x],5,5,100)

print(time.time()-s)                  


#DF.to_csv('/Users/lukishyadav/Desktop/Gittable_Work/Commuters/100_Customers_CLustered.csv',index=False)
 
"""
We have the label and e_label in dataframe


records having both label and e_label are rentals with commute between clusters i.e rental counted as commuters rental.


"""

#len(DF)

"""
Commuters rentals
"""

DDD=DF[(DF['label']!='-1') & (DF['e_label']!='-1')]


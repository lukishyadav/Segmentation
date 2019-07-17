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
from bokeh.tile_providers import CARTODBPOSITRON 
import numpy as np
from sklearn.cluster import DBSCAN 
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider,TextInput

df=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Commuters/commuters.csv')



def convert_to_mercator(lngs, lats):
    projection = Proj(init='epsg:3857')
    xs = []
    ys = []
    for lng, lat in zip(lngs, lats):
        x, y = projection(lng, lat)
        xs.append(x)
        ys.append(y)
    return xs, ys


df['try']=df['start location lat/lng'].apply(lambda x:x.split(","))

df['lat_col']=df['try'].apply(lambda x:float(x[0]))

df['lng_col']=df['try'].apply(lambda x:float(x[1]))


df['merc_lng'],df['merc_lat']=convert_to_mercator(df['lng_col'],df['lat_col'])


maxlat=max(df['merc_lat'])
minlat=min(df['merc_lat'])

maxlng=max(df['merc_lng'])
minlng=min(df['merc_lng'])


kms_per_radian = 6371.0088


global min_meters 

def set_clusters(df, percent, min_n, min_meters):
    # set the data up for clustering
    min_percent = int(df.shape[0]/100) * percent  # 1% of samples available
    epsilon = float(min_meters)/1000/kms_per_radian  # m * (1km/1000m) * (radian/km) => radians
    X = np.column_stack((df['lng_col'], df['lat_col']))
    # clustering
    dbscan = DBSCAN(eps=epsilon, min_samples=max(min_percent, min_n),
                    algorithm='ball_tree',
                    metric='haversine').fit(np.radians(X))

    # create list of clusters and labels
    n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    n_noise_ = list(dbscan.labels_).count(-1)

    # label the points unique to their cluster
    unique_labels = set(dbscan.labels_)
    df['label'] = [str(label) for label in dbscan.labels_]

    return df







def my_slider_handler(attr,old,new):
    slider.value=new
    min_meters=slider.value
    #   source.change.emit()
    filtered_df=set_clusters(df[df['customer_id']==21],5,5,min_meters)
    noise_df = filtered_df[filtered_df['label'] == '-1']
    datapoints_df = filtered_df[filtered_df['label'] != '-1']
    #noise_source = ColumnDataSource()
    #datapoints_source = ColumnDataSource()
    
    # modify columndatasources to modify the figures
    noise_source.data = dict(
        x=noise_df['merc_lng'],
        y=noise_df['merc_lat'],
        label=noise_df['label'],
        #time=noise_df['start_datetime']
        )
    datapoints_source.data = dict(
        x=datapoints_df['merc_lng'],
        y=datapoints_df['merc_lat'],
        label=datapoints_df['label'],
        #time=datapoints_df['start_datetime']
        )


# create some widgets
slider = Slider(start=50, end=10000, value=50, step=50, title="Epsilon")
slider.on_change("value", my_slider_handler)

min_meters=50

filtered_df=set_clusters(df[df['customer_id']==21],5,5,min_meters)

check=df[df['customer_id']==21]

# separate out the noise and the clustered points
noise_df = filtered_df[filtered_df['label'] == '-1']
datapoints_df = filtered_df[filtered_df['label'] != '-1']


noise_source = ColumnDataSource()
datapoints_source = ColumnDataSource()

# modify columndatasources to modify the figures
noise_source.data = dict(
    x=noise_df['merc_lng'],
    y=noise_df['merc_lat'],
    label=noise_df['label'],
    #time=noise_df['start_datetime']
    )
datapoints_source.data = dict(
    x=datapoints_df['merc_lng'],
    y=datapoints_df['merc_lat'],
    label=datapoints_df['label'],
    #time=datapoints_df['start_datetime']
    )



palette_range = [str(x) for x in range(0, 20)]
from bokeh.palettes import Category20

map_repr='mercator'


# set up/draw the map
map_figure = figure(
    x_range=(minlng,maxlng),
    y_range=(minlat, maxlat),
    x_axis_type=map_repr,
    y_axis_type=map_repr,
    title='Clustering Map Representation'
)
map_figure.add_tile(CARTODBPOSITRON)

#show(map_figure)

  
def plot_points(map_figure, noise_source, datapoints_source):
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
                          factors=palette_range),line_color='black')
                      
    #show(map_figure)                  
                      
 

                     
plot_points(map_figure, noise_source, datapoints_source) 

inputs=column(row(widgetbox(slider, width=600)),row(map_figure),width=1000)

layout = column(
            row(
                widgetbox(slider, width=600),
                width=1000),    
            row(map_figure),
        )

curdoc().add_root(layout)
curdoc().title = 'DOW and Hour Clustering Analysis'

#logging.info('initial map drawn')


#runfile('C:/yourfolder/myfile.py',args='one two three')




                  


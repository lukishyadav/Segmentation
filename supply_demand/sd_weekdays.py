#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:12:34 2019

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
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider,TextInput,RadioGroup
from bokeh.palettes import Category20


df=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/supply_demand/supply_demand_counts_20190501_20190606.csv')


DF=df.copy()


['Unnamed: 0', 'Date', 'Appopen_Count', 'Supply_Count', 'Rental_Count',
       'hour']


"""

Return the day of the week as an integer, where Monday is 0 and Sunday is 6.

"""

from datetime import datetime

DF['Day']=DF['Date'].apply(lambda x:datetime.strptime(x[0:10],'%Y-%m-%d'))

DF['Date']=DF['Date'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))


DF['month']=DF['Date'].apply(lambda x:x.month)
#DF['Weekday']=DF['Date'].apply(lambda x:x.weekday())
#DF['Weekday']=DF['Weekday'].astype(str)

DF['Weekday']=DF['Day'].apply(lambda x:x.strftime("%A"))

LL=list(set(DF['Weekday']))

DDD=DF.groupby(['month'])['Rental_Count'].sum()
DDD=DDD.to_frame()
DDD.reset_index(inplace=True)
#DD=DF.groupby(['Day']).size().reset_index(name='counts')


#['Unnamed: 0', 'Date', 'Appopen_Count', 'Supply_Count', 'Rental_Count','hour']

DD=DF.groupby(['Day']).sum()

DD=DF.groupby(['Day'])['Rental_Count'].sum()
#DD.reset_index(inplace=True)
DD=DD.to_frame()
DD.reset_index(inplace=True)
#D['Weekday']=DD['Day'].apply(lambda x:x.weekday())
DD['Weekday']=DD['Day'].apply(lambda x:x.strftime("%A"))


#DD['Weekday']=DD['Weekday'].astype(str)
#['Day', 'Rental_Count', 'Weekday']

def my_radio_handler(new):
    if new==0:
        data.data = dict(
        x=DF['Date'],
        y=DF['Rental_Count'],
        label=DF['Weekday'],
    #time=noise_df['start_datetime']
    )
        
    elif new==1:
                data.data = dict(
                x=DD['Day'],
                y=DD['Rental_Count'],
                label=DD['Weekday'],
            #time=noise_df['start_datetime']
    )

    else:  
                data.data = dict(
                x=DDD['month'],
                y=DDD['Rental_Count'],
                label=DDD['Weekday'],
            #time=noise_df['start_datetime']
    )
    


radio_group = RadioGroup(
    labels=["hourly", "daily", "monthly"], active=1)
radio_group.on_click(my_radio_handler)

data= ColumnDataSource()
data.data = dict(
    x=DD['Day'],
    y=DD['Rental_Count'],
    label=DD['Weekday'],
    #time=noise_df['start_datetime']
    )

#p = figure(plot_width=1200, plot_height=400)

p = figure(plot_width=1200, plot_height=400,x_axis_type='datetime')

#palette_range = [str(x) for x in range(0, 20)]

palette_range = LL

p.circle(x='x', y='y', size=6,
                      fill_alpha=1, source=data,
                      fill_color=factor_cmap(
                          'label',
                          palette=Category20[20],
                          factors=palette_range),line_color='black',legend='label')


#show(p)

layout = row(
            column(
                widgetbox(radio_group,width=350),
                width=400),    
            column(p,width=800),
        )

curdoc().add_root(layout)
curdoc().title = 'Supply_Demand_Weekdays'

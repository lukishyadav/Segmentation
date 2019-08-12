#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:12:34 2019

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


df=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/supply_demand/supply_demand_counts_20190501_20190606.csv')


DF=df.copy()

%matplotlib auto
import pandas
import matplotlib.pyplot as plt

plt.plot(DF['Rental_Count'])

['Unnamed: 0', 'Date', 'Appopen_Count', 'Supply_Count', 'Rental_Count',
       'hour']


train=DF['Supply_Count'].values

from pyramid.arima import auto_arima


model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)


forecast = model.predict(n_periods=10)


DF['Date'].iloc[1].today().weekday()


"""

Return the day of the week as an integer, where Monday is 0 and Sunday is 6.

"""

from datetime import datetime

DF['Day']=DF['Date'].apply(lambda x:datetime.strptime(x[0:10],'%Y-%m-%d'))

DF['Date']=DF['Date'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

DF['Weekday']=DF['Date'].apply(lambda x:x.weekday())


DD=DF.groupby(['Day']).size().reset_index(name='counts')

#['Unnamed: 0', 'Date', 'Appopen_Count', 'Supply_Count', 'Rental_Count','hour']

DD=DF.groupby(['Day']).sum()

DD=DF.groupby(['Day'])['Rental_Count'].sum()
#DD.reset_index(inplace=True)
DD=DD.to_frame()
DD.reset_index(inplace=True)
DD['Weekday']=DD['Day'].apply(lambda x:x.weekday())
#DD['Weekday']=DD['Day'].apply(lambda x:x.strftime("%A"))


DD['Weekday']=DD['Weekday'].astype(str)
#['Day', 'Rental_Count', 'Weekday']

data= ColumnDataSource()
data.data = dict(
    x=DD['Day'],
    y=DD['Rental_Count'],
    label=DD['Weekday'],
    #time=noise_df['start_datetime']
    )

p = figure(plot_width=1200, plot_height=400,x_axis_type='datetime')

palette_range = [str(x) for x in range(0, 20)]

p.circle(x='x', y='y', size=6,
                      fill_alpha=1, source=data,
                      fill_color=factor_cmap(
                          'label',
                          palette=Category20[20],
                          factors=palette_range),line_color='black',legend='label')


show(p)



%matplotlib auto
import pandas
import matplotlib.pyplot as plt

plt.plot(DD['Rental_Count'])



DF['Date'].iloc[31]

from numpy import array

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)

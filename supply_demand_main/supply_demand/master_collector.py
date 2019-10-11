import time
st=time.time()


import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/Users/lukishyadav/Desktop/segmentation/supply_demand_main/codes/flow')

import sd_module as sd

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
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
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot
from keras.models import load_model
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
import os
from matplotlib import pyplot as plt


#df=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/data_big/data/quadrant_0/timescale_90/hex_edge_24.911m_quantile_3_daily.csv')
#174.376,  461.355, 1220.63

[174.376,461.355,1220.63]

#file='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/data/supply/darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_quantile_3_hourly.csv'

file='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/data/demand/darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_all_hexes_hourly.csv'


df=pd.read_csv(file)

#Supply_Demand_Data_Sync
dfd=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/data/supply/darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_all_hexes_hourly.csv')

CL=df.columns
CL2=dfd.columns
CL3=set(CL).intersection(CL2)
df=pd.merge(df['timeseries'],dfd[CL3],on='timeseries',how='inner')

cals=list(CL3)
cals.sort(reverse=True)
df=df[cals]

import re
result = re.search('ge_(.*)_hourly', file)
print(result.group(1))

qua=result.group(1)

store=df.columns
mname=file[-35:-4]


LL=len(df.columns)

LLL=[str(i) for i in range((LL-1))]

df.columns=['date']+LLL
master=pd.DataFrame(data=np.array([[0 for i in range(len(LLL))],[0 for i in range(len(LLL))]]).T,columns=['Hex','Supply-Demand_Prediction'])
#key=len(LLL)-2
#key=0
mpath='/Users/lukishyadav/Desktop/sd_result' 
for key in LLL:
    
    
    #461.355m_all_hexes_key_0_38.00287838510444, -122.30510401783533
    dpath='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/images/demand/'+result.group(1)+'_key_'+str(key)+'_'+store[int(key)+1][1:-1]
    spath='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/images/supply/'+result.group(1)+'_key_'+str(key)+'_'+store[int(key)+1][1:-1]

    dframe=pd.read_csv(dpath+'/OutputFile_Key_'+str(key)+'.csv')
    sframe=pd.read_csv(spath+'/OutputFile_Key_'+str(key)+'.csv')
    
    """Index(['Unnamed: 0', 'Best RMSE for windows', 'Best Model for Windows',
       'Prediction of best model', 'All Predicted Outputs'],
      dtype='object')"""
    
    drmse=eval(dframe['Best RMSE for windows'].iloc[0])
    
    d_win= min(drmse, key=drmse.get)
    d_best_prediction=eval(dframe['Prediction of best model'].iloc[0])
    d_best_prediction=d_best_prediction[d_win]
    d_best_model=eval(dframe['Best Model for Windows'].iloc[0])[d_win]
    d_original=eval(dframe['Original'].iloc[0])['o']

    srmse=eval(sframe['Best RMSE for windows'].iloc[0])
    
    s_win= min(srmse, key=srmse.get)
    s_best_prediction=eval(sframe['Prediction of best model'].iloc[0])
    s_best_prediction=s_best_prediction[s_win]
    s_best_model=eval(sframe['Best Model for Windows'].iloc[0])[s_win]
    s_original=eval(sframe['Original'].iloc[0])['o']
    
    import matplotlib.pyplot as plt
    plt.plot(s_original[0],label='Supply Original')
    plt.plot(s_best_prediction,label='Supply Prediction ('+s_best_model+')')
    plt.plot(d_original[0],label='Demand Original')
    plt.plot(d_best_prediction,label='Demand Prediction ('+d_best_model+')')
    #plt.plot(aa_forecast,label='AutoArima Forecast')
    #plt.plot(foreca,label='Float AutoArima Forecast')
    plt.legend()
    plt.title('Supply-Demand_Prediction')
    #plt.xlabel('Progression in '+str(metric))
    plt.xlabel('Progression in '+'quarter day')
    plt.ylabel('Value')
    plt.savefig('Image'+str(key)+'.png')
    plt.clf()
    
    #master=pd.DataFrame(columns=['Hex','Supply-Demand_Prediction'])
    ppath='Image'+str(key)+'.png'
    master['Supply-Demand_Prediction'].iloc[int(key)]='<img src="{}" /> '.format(ppath)
    master['Hex'].iloc[int(key)]='Hex_no:'+str(key)+' LL_value:'+store[int(key)+1]
    #master.to_html(dpath+'/test.html', escape=False)
    #plt.savefig(dpath+'/Final_Output_window'+str(mkey)+'.png')

#mpath='/Users/lukishyadav/Desktop/sd_result'  
master.to_html('test.html', escape=False)
    

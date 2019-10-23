import time
st=time.time()


import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/Users/lukishyadav/Desktop/segmentation/supply_demand_main/codes/flow')

#import sd_module as sd

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

#Supply_Demand_Data_Sync
dfd=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/data/demand/darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_all_hexes_hourly.csv')

dfs=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/data/supply/darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_all_hexes_hourly.csv')

CL=dfs.columns
CL2=dfd.columns
#CL4=DF2.columns
CL3=set(CL2).intersection(CL)





file='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/app_open_demand/known_trips_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_all_hexes_hourly.csv'


df=pd.read_csv(file)

#Supply_Demand_Data_Sync
dfu=pd.read_csv('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/app_open_demand/unknown_trips_breakdown/quadrant_0/timescale_30/hex_edge_461.355m_all_hexes_hourly.csv')

for n in CL3: 
    if n not in df.columns:
        df[n]=[0 for n in range(len(df))]
    elif n not in dfu.columns:
        dfu[n]=[0 for n in range(len(dfu))]

DF=pd.merge(df[CL3],dfu[CL3],on='timeseries',how='outer')

DF.fillna(0,inplace=True)

# =============================================================================
# CL=df.columns
# CL2=dfu.columns
# CL3=set(CL2).intersection(CL)
# 
# DF=pd.merge(df[CL3],dfu[CL3],on='timeseries',how='inner')
# 
# =============================================================================
CL3.remove('timeseries')

for x in CL3:
    DF[x]=DF[x+'_x']+DF[x+'_y']


CL3.add('timeseries')
DF2=DF[CL3]



cals=list(CL3)
cals.sort(reverse=True)
DF2=DF2[cals]


df=DF2.copy()

import re
result = re.search('ge_(.*)_hourly', file)
print(result.group(1))

qua=result.group(1)

store=df.columns
mname=file[-35:-4]


LL=len(df.columns)

LLL=[str(i) for i in range((LL-1))]

df.columns=['date']+LLL


master=pd.DataFrame(data=np.array([[0 for i in range(len(LLL))],[0 for i in range(len(LLL))],[0 for i in range(len(LLL))],[0 for i in range(len(LLL))],[0 for i in range(len(LLL))]]).T,columns=['Hex','Supply-Demand_Prediction','Original Data','OAP_points','Evaluation'])
#key=len(LLL)-2
#key=0
#LLL=[str(i) for i in [36,34,29,27,24,22,15]]


mpath='/Users/lukishyadav/Desktop/sd_result' 
for key in LLL:
    
    
    #461.355m_all_hexes_key_0_38.00287838510444, -122.30510401783533
    dpath='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/images/demand/'+result.group(1)+'_key_'+str(key)+'_'+store[int(key)+1][1:-1]
    spath='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/images/supply/'+result.group(1)+'_key_'+str(key)+'_'+store[int(key)+1][1:-1]
    adpath='/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/images/app_open_demand/'+result.group(1)+'_key_'+str(key)+'_'+store[int(key)+1][1:-1]
    dframe=pd.read_csv(dpath+'/OutputFile_Key_'+str(key)+'.csv')
    sframe=pd.read_csv(spath+'/OutputFile_Key_'+str(key)+'.csv')
    adframe=pd.read_csv(adpath+'/OutputFile_Key_'+str(key)+'.csv')
    
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
    
    adrmse=eval(adframe['Best RMSE for windows'].iloc[0])
    
    ad_win= min(adrmse, key=adrmse.get)
    ad_best_prediction=eval(adframe['Prediction of best model'].iloc[0])
    ad_best_prediction=ad_best_prediction[ad_win]
    ad_best_model=eval(adframe['Best Model for Windows'].iloc[0])[ad_win]
    ad_original=eval(adframe['Original'].iloc[0])['o']
    
    from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
    from math import sqrt
    mse=mean_squared_error(ad_original[0], ad_best_prediction)
    rmse=sqrt(mean_squared_error(ad_original[0], ad_best_prediction))
    mae=mean_absolute_error(ad_original[0], ad_best_prediction)
    r2=r2_score(ad_original[0], ad_best_prediction)
    
    def a_original(path):
        ffc=pd.read_csv(path+'/original_data_'+str(key)+'.csv')
        ffc['original_data']
        all_original=eval(ffc['original_data'].iloc[0])
        return all_original[ad_win]
    
    d_all_o=a_original(dpath)
    s_all_o=a_original(spath)
    ad_all_o=a_original(adpath)
    
    import matplotlib.pyplot as plt
    
    """
    plt.plot(s_original[0],label='Supply Original')
    plt.plot(s_best_prediction,label='Supply Prediction ('+s_best_model+')'+'Best_Window :'+str(s_win))
    plt.plot(d_original[0],label='Demand Original')
    plt.plot(d_best_prediction,label='Demand Prediction ('+d_best_model+')'+'Best_Window :'+str(d_win))
    plt.plot(ad_original[0],label='App-Open Demand Original')
    plt.plot(ad_best_prediction,label='App-Open Demand Prediction ('+ad_best_model+')'+'Best_Window :'+str(ad_win))
    #plt.plot(aa_forecast,label='AutoArima Forecast')
    #plt.plot(foreca,label='Float AutoArima Forecast')
    plt.legend()
    plt.title('Supply-Demand_Prediction')
    #plt.xlabel('Progression in '+str(metric))
    plt.xlabel('Progression in '+'quarter day')
    plt.ylabel('Value')
    plt.savefig('Image'+str(key)+'.png')
    plt.clf()
    """
    

    plt.plot(ad_original[0],label='AODO')
    plt.plot(ad_best_prediction,label='AODOP ('+ad_best_model+')'+'BW :'+str(ad_win))
    plt.legend()
    plt.xlabel('Progression in '+'quarter day')
    plt.ylabel('Value')
    
    plt.savefig('Image'+str(key)+'.png')
    plt.clf()
    
# =============================================================================
#     import matplotlib.pyplot as plt
#     plt.plot(d_all_o,label='Supply Original Window :'+str(s_win))
#     plt.plot(s_all_o,label='Demand Original Window :'+str(d_win))
#     plt.plot(ad_all_o,label='App-open Demand Original Window :'+str(ad_win))
#       #plt.plot(aa_forecast,label='AutoArima Forecast')
#     #plt.plot(foreca,label='Float AutoArima Forecast')
#     plt.legend()
#     plt.title('Original_Data')
#     #plt.xlabel('Progression in '+str(metric))
#     plt.xlabel('Progression in '+'quarter day')
#     plt.ylabel('Value')
#     plt.savefig('Original'+str(key)+'.png')
#     plt.clf()
# =============================================================================
    
    
    plt.plot(ad_all_o,label='App-open Demand Original Window :'+str(ad_win))
    plt.legend()
    plt.xlabel('Progression in '+'quarter day')
    plt.ylabel('Value')
    plt.savefig('Original'+str(key)+'.png')
    plt.clf()
    #plt.show()
    
    
    #master=pd.DataFrame(columns=['Hex','Supply-Demand_Prediction'])
    ppath='Image'+str(key)+'.png'
    opath='Original'+str(key)+'.png'
    master['Supply-Demand_Prediction'].iloc[int(key)]='<img src="{}" /> '.format(ppath)
    master['Original Data'].iloc[int(key)]='<img src="{}" /> '.format(opath)
    master['OAP_points'].iloc[int(key)]=str({'Og':list(ad_original[0]),'Pred':list(ad_best_prediction)})
    master['Evaluation'].iloc[int(key)]=str({'rmse':rmse,'mse':mse,'mae':mae,'r2':r2})   
    master['Hex'].iloc[int(key)]='Hex_no:'+str(key)+' LL_value:'+store[int(key)+1]
    #master.to_html(dpath+'/test.html', escape=False)
    #plt.savefig(dpath+'/Final_Output_window'+str(mkey)+'.png')

#mpath='/Users/lukishyadav/Desktop/sd_result'  
master.to_html('test.html', escape=False)
 
master['rmse']=master['Evaluation'].apply(lambda x:eval(x)['rmse'])
master['mse']=master['Evaluation'].apply(lambda x:eval(x)['mse'])
master['mae']=master['Evaluation'].apply(lambda x:eval(x)['mae'])
master['r2']=master['Evaluation'].apply(lambda x:eval(x)['r2'])



CCCC=['Hex', 'Supply-Demand_Prediction', 'Original Data', 'OAP_points',
       'Evaluation']

df_rmse=master.sort_values(['rmse'], ascending=True)
df_rmse.to_html('rmse.html', escape=False)
df_mse=master.sort_values(['mse'], ascending=True)
df_mse.to_html('mse.html', escape=False)

df_mae=master.sort_values(['mae'], ascending=True)
df_mae.to_html('mae.html', escape=False)

master = master[(master.T != 0).any()]   

master.to_html('test.html', escape=False)


"""
Decent results

FPR APP OPEN DEMAND

36,34,29,27,24,22,15,

"""

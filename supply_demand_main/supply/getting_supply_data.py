#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:08:06 2019

@author: lukishyadav
"""

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/Users/lukishyadav/Desktop/segmentation/supply_demand_main/supply')

import calendar
from copy import deepcopy
import time

import pandas as pd
import numpy as np
from settings import region
from datetime import datetime


VEHICLE_DATAFILE='supply.csv'
miles_per_meter = 0.000621371


#df=pd.read_csv('supply.csv')
"""

mi = pd.MultiIndex.from_product([list(calendar.day_name), list(range(0, 24))], names=['dow', 'hour'])



base_series = pd.Series(index=mi).fillna(value=0)

mi_df = pd.DataFrame(columns=mi)
supply_df = pd.DataFrame()
"""

REGION_TIMEZONE = region['oakland']['timezone']

def convert_datetime_columns(df, columns):
    for col in columns:
        try:
            df[col] = df[col].dt.tz_localize('UTC').dt.tz_convert(REGION_TIMEZONE)
        except TypeError:
            df[col] = df[col].dt.tz_convert(
                   'UTC').dt.tz_convert(REGION_TIMEZONE)
            
            
            
VEHICLE_DT_COLS = ['event_datetime']

print('Loading file...')
read_init_time = time.time()

# get df and clean up
vehicle_df = pd.read_csv(
    VEHICLE_DATAFILE,
    parse_dates=VEHICLE_DT_COLS,
    infer_datetime_format=True
).dropna()
print('Done loading file...')
print(f'Took {int(time.time() - read_init_time)}s')


# Replacing reservation status by is_available

vehicle_df['is_available']=vehicle_df['status'].apply(lambda x:True if x=='free' else False)






print('Converting datetime columns...')
convert_datetime_columns(vehicle_df, VEHICLE_DT_COLS)
print('Done converting datetime columns...')


FD=str(vehicle_df['event_datetime'].iloc[0])
LD=str(vehicle_df['event_datetime'].iloc[-1])

FD = datetime.strptime(FD[0:13], '%Y-%m-%d %H')
LD = datetime.strptime(LD[0:13], '%Y-%m-%d %H')

years=list(set(pd.date_range(FD, LD, freq='H', closed='left').year))
months=list(set(pd.date_range(FD, LD, freq='H', closed='left').month))
days=list(set(pd.date_range(FD, LD, freq='H', closed='left').day))
hours=list(set(pd.date_range(FD, LD, freq='H', closed='left').hour))

DL=list(pd.date_range(FD, LD, freq='H', closed='left'))

DL=list(pd.date_range(FD, LD, freq='H'))


# group by vin
vehicle_df_vin_grouped = vehicle_df.groupby(['vin'])
            

DF=vehicle_df[vehicle_df['vin']=='JTDKDTB36H1592743']


mi = pd.MultiIndex.from_product([DL], names=['datetime'])



base_series = pd.Series(index=mi).fillna(value=0)

mi_df = pd.DataFrame(columns=mi)
supply_df = pd.DataFrame()

"""

group=DF.copy()

group = group.sort_values(by='event_datetime')

left = group[(group['is_available'] == False) & (group['is_available'].shift() == True)].rename(
        columns={'event_datetime':'unavailable_at'})

right = group[(group['is_available'] == True) & (group['is_available'].shift() == False)].rename(
        columns={'event_datetime':'available_at'})['available_at'].to_frame()


merged_group = pd.merge_asof(left, right, left_on='unavailable_at', right_on='available_at')

"""



def collapse_is_available_events(group):
    global supply_df
    group = group.sort_values(by='event_datetime')

    # get time of change of states

    # get event_datetime when is_available goes from true to false (becomes unavailable)
    # previous event (is_available=True) changed state (is_available=False), indicating becoming unavailable
    left = group[(group['is_available'] == False) & (group['is_available'].shift() == True)].rename(
        columns={'event_datetime':'unavailable_at'})

    # get event_datetime when is_available goes from false to true (becomes available)
    # previous event (is_available=False) changed state (is_available=True), indicating becoming available
    right = group[(group['is_available'] == True) & (group['is_available'].shift() == False)].rename(
        columns={'event_datetime':'available_at'})['available_at'].to_frame()

    # can't assume symmetry for events
    # can't tell which event comes first
    merged_group = pd.merge_asof(left, right, left_on='unavailable_at', right_on='available_at')
    supply_df = supply_df.append(merged_group)

    global vehicle_df_vin_grouped

    if not supply_df.shape[0] % 1000:
        print(f'{supply_df.shape[0]} events collapsed')
        
        
# construct large dow/hour df
# NOTE: very expensive. should save intermediates so don't have to regenerate
def extractor(x):
    global mi_df
    temp = deepcopy(base_series)
    # duration less than 1 hour, does span across slice (hour) ex: [1:30, 2:15]
    if x.size == 2 and x[0].hour != x[1].hour:
        temp[x[0].day_name(), x[0].hour] += 60 - x[0].minute
        temp[x[1].day_name(), x[1].hour] += x[1].minute

    # duration less than 1 hour, doesn't span across slice (hour) ex: [1:30, 1:45]
    elif x.size == 2 and x[0].hour == x[1].hour:
        temp[x[0].day_name(), x[0].hour] += x[1].minute - x[0].minute

    # duration greater than 1 hour, does span across slice (hour) ex: [1:30, 2:30, 2:45]
    elif x.size == 3 and x[1].hour == x[2].hour:
        temp[x[0].day_name(), x[0].hour] += 60 - x[0].minute
        temp[x[2].day_name(), x[2].hour] += x[2].minute

    else:
        # duration greater than 2 hours, ex: [1:30, 2:30, 3:30, 3:45]
        # or spans across multiple hours
        n = 0
        min_marker = x[0].minute
        for i, j, k in zip(x.day_name(), x.hour, x.minute):
            # each datetimeindex
            if n == 0: # first element => 60 - 30 = 30
                temp[i, j] += (60 - k)
            elif n == (x.size - 1):  # last element, can't assume full hour
                if k >= min_marker:
                    temp[i, j] += (k - min_marker) # ex: 3:45 - 3:30 = 15m
                else:
                    temp[i, j] += k  # ex: 3:30 - 3:00 = 30m
            elif n == (x.size - 2):  # second to last element, can't assume full hour
                temp[i, j] += k  # ex: 3:30 - 3:00 = 30m
            else:  # middle of array
                temp[i, j] += 60 # ex: 3:30 - 2:30 = 1h
            n += 1
    mi_df = mi_df.append(temp, ignore_index=True)
    # get size incoming vehicle events
    global df
    # determine size of mi_df
    # report every 10000 events
    if not mi_df.shape[0] % 10000:
        print(f'mask {mi_df.shape[0]/df.shape[0] * 100}% complete')


def extractor2(x):
    global mi_df
    temp = deepcopy(base_series)
    # duration less than 1 hour, does span across slice (hour) ex: [1:30, 2:15]
    if x.size == 2 and x[0].hour != x[1].hour:
        temp[datetime.strptime(str(x[0])[0:13], '%Y-%m-%d %H')] += 60 - x[0].minute
        temp[datetime.strptime(str(x[1])[0:13], '%Y-%m-%d %H')] += x[1].minute

    # duration less than 1 hour, doesn't span across slice (hour) ex: [1:30, 1:45]
    elif x.size == 2 and x[0].hour == x[1].hour:
        temp[datetime.strptime(str(x[0])[0:13], '%Y-%m-%d %H')] += x[1].minute - x[0].minute

    # duration greater than 1 hour, does span across slice (hour) ex: [1:30, 2:30, 2:45]
    elif x.size == 3 and x[1].hour == x[2].hour:
        temp[datetime.strptime(str(x[0])[0:13], '%Y-%m-%d %H')] += 60 - x[0].minute
        temp[datetime.strptime(str(x[2])[0:13], '%Y-%m-%d %H')] += x[2].minute

    else:
        # duration greater than 2 hours, ex: [1:30, 2:30, 3:30, 3:45]
        # or spans across multiple hours
        n = 0
        min_marker = x[0].minute
        for i, k in zip(x, x.minute):
            # each datetimeindex
            if n == 0: # first element => 60 - 30 = 30
                temp[datetime.strptime(str(i)[0:13], '%Y-%m-%d %H')] += (60 - k)
            elif n == (x.size - 1):  # last element, can't assume full hour
                if k >= min_marker:
                    temp[datetime.strptime(str(i)[0:13], '%Y-%m-%d %H')] += (k - min_marker) # ex: 3:45 - 3:30 = 15m
                else:
                    temp[datetime.strptime(str(i)[0:13], '%Y-%m-%d %H')] += k  # ex: 3:30 - 3:00 = 30m
            elif n == (x.size - 2):  # second to last element, can't assume full hour
                temp[datetime.strptime(str(i)[0:13], '%Y-%m-%d %H')] += k  # ex: 3:30 - 3:00 = 30m
            else:  # middle of array
                temp[datetime.strptime(str(i)[0:13], '%Y-%m-%d %H')] += 60 # ex: 3:30 - 2:30 = 1h
            n += 1
    mi_df = mi_df.append(temp, ignore_index=True)
    # get size incoming vehicle events
    global df
    # determine size of mi_df
    # report every 10000 events
    if not mi_df.shape[0] % 10000:
        print(f'mask {mi_df.shape[0]/df.shape[0] * 100}% complete')






print('Collapsing events...')
collapse_init_time = time.time()
vehicle_df_vin_grouped.apply(collapse_is_available_events)
print(f'Collapse time: {time.time() - collapse_init_time}s')

supply_df = supply_df.dropna()
supply_df.reset_index(inplace=True)
supply_df['idle_duration'] = supply_df['unavailable_at'] - supply_df['available_at']  # duration for analysis
supply_df['idle_duration_minutes'] = supply_df['idle_duration'].dt.total_seconds()/60.0

# create datetimeindex of periods with the end datetime appended
df = supply_df.apply(
    lambda x: (pd.date_range(x['available_at'], x['unavailable_at'], freq='H', closed='left')).append(
        pd.to_datetime([x['unavailable_at']])), axis=1)

print('Extracting events...')
extract_init_time = time.time()
df.apply(extractor2)
print(f'Extract time: {time.time() - extract_init_time}s')

# merge the big dow/hour mask back with vehicle_update data
supply_df = supply_df.merge(mi_df, left_index=True, right_index=True)
supply_df.to_csv(f'{VEHICLE_DATAFILE.split(".")[0]}_with_datetime_mask.csv')   

kd=supply_df.copy()
kd.drop('index',axis=1,inplace=True)
#kd.stack()

unaltered=['unavailable_at',
 'vin',
 'is_available',
 'lat',
 'lng',
 'status',
 'available_at',
 'idle_duration',
 'idle_duration_minutes']
KD=kd.copy()

KD.drop(columns=unaltered,inplace=True)
alter=list(KD.columns)

#Melting of Dataframe

df2=pd.melt(kd,id_vars=unaltered,var_name='start_datetime', value_name='Minutes')

df2.to_csv('Supply_Data.csv',index=False)

#df2.to_csv('Supply_Data.csv',index=False,date_format='%Y-%m-%d %H')


pd.date_range(supply_df['available_at'].iloc[1], supply_df['unavailable_at'].iloc[1], freq='H', closed='left')     


datetime.strptime(str(x[0])[0:13], '%Y-%m-%d %H')




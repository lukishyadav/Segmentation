#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:04:48 2019

@author: lukishyadav
"""
import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/Users/lukishyadav/Desktop/Segmentation/supply_demand_main/supply_demand/app_open_demand')
import time
import pytz

import pandas as pd
import numpy as np

from settings import region

miles_per_meter = 0.000621371

selected_region = region['oakland']
REGION_TIMEZONE = selected_region['timezone']


# converts incoming data to proper timezone
def convert_datetime_columns(df, columns):
    for col in columns:
        try:
            df[col] = df[col].dt.tz_localize('UTC').dt.tz_convert(REGION_TIMEZONE)
        except TypeError:
            df[col] = df[col].dt.tz_convert(
                   'UTC').dt.tz_convert(REGION_TIMEZONE)

# constrains to a bounding box
def set_bbox(df, lng_min, lat_min, lng_max, lat_max):
#     df = df[(df['lat'] >= region['lat_min']) & (df['lat'] <= region['lat_max'])]
#     df = df[(df['lng'] >= region['lng_min']) & (df['lng'] <= region['lng_max'])]
    df = df[(df['lat'] >= lat_min) & (df['lat'] <= lat_max)]
    df = df[(df['lng'] >= lng_min) & (df['lng'] <= lng_max)]
    return df

# import the dataset
raw_rental_datafile = 'known_trips.csv'
raw_rental_df = pd.read_csv(
        raw_rental_datafile,
        parse_dates=['created_at'],
        infer_datetime_format=True
    ).dropna()

raw_rental_df.rename(columns={'created_at':'start_datetime'},inplace=True)

# remove extraneous datapoints
raw_rental_df = set_bbox(raw_rental_df, selected_region['lng_min'], selected_region['lat_min'],
                         selected_region['lng_max'], selected_region['lat_max'])

# perform all the extracts and data transformation prior to sorting dataset
# extract the rental start dow/hour
raw_rental_df['start_datetime_hour'] = raw_rental_df['start_datetime'].dt.hour
raw_rental_df['start_datetime_dow'] = raw_rental_df['start_datetime'].dt.day_name()
raw_rental_df['start_date'] = raw_rental_df['start_datetime'].dt.date

quadrant_dataset = [
    set_bbox(raw_rental_df, selected_region['lng_center'], selected_region['lat_center'], selected_region['lng_max'], selected_region['lat_max']), # I
    set_bbox(raw_rental_df, selected_region['lng_min'], selected_region['lat_center'], selected_region['lng_center'], selected_region['lat_max']), # II
    set_bbox(raw_rental_df, selected_region['lng_min'], selected_region['lat_min'], selected_region['lng_center'], selected_region['lat_center']), # III
    set_bbox(raw_rental_df, selected_region['lng_center'], selected_region['lat_min'], selected_region['lng_max'], selected_region['lat_center'])  # IV
]


def draw_quadrants_map():
    import folium
    quadrants_map = folium.Map(location=[selected_region['lat_center'], selected_region['lng_center']])
    folium.PolyLine([
        (selected_region['lat_min'], selected_region['lng_min']),
        (selected_region['lat_min'], selected_region['lng_max'])]).add_to(quadrants_map) # bottom line
    folium.PolyLine([
        (selected_region['lat_max'], selected_region['lng_min']),
        (selected_region['lat_max'], selected_region['lng_max'])]).add_to(quadrants_map) # top line
    folium.PolyLine([
        (selected_region['lat_min'], selected_region['lng_max']),
        (selected_region['lat_max'], selected_region['lng_max'])]).add_to(quadrants_map) # right line
    folium.PolyLine([
        (selected_region['lat_min'], selected_region['lng_min']),
        (selected_region['lat_max'], selected_region['lng_min'])]).add_to(quadrants_map) # left line
    folium.PolyLine([
        (selected_region['lat_center'], selected_region['lng_min']),
        (selected_region['lat_center'], selected_region['lng_max'])]).add_to(quadrants_map) # center line, vertical
    folium.PolyLine([
        (selected_region['lat_min'], selected_region['lng_center']),
        (selected_region['lat_max'], selected_region['lng_center'])]).add_to(quadrants_map) # center line horizontal

    return quadrants_map

draw_quadrants_map()


from h3 import h3

max_res = 15
list_hex_edge_km = []
list_hex_edge_m = []
list_hex_perimeter_km = []
list_hex_perimeter_m = []
list_hex_area_sqkm = []
list_hex_area_sqm = []

for i in range(0,max_res + 1):
    ekm = h3.edge_length(resolution=i, unit='km')
    em = h3.edge_length(resolution=i, unit='m')
    list_hex_edge_km.append(round(ekm,3))
    list_hex_edge_m.append(round(em,3))
    list_hex_perimeter_km.append(round(6 * ekm,3))
    list_hex_perimeter_m.append(round(6 * em,3))
    
    akm = h3.hex_area(resolution=i, unit='km^2')
    am = h3.hex_area(resolution=i, unit='m^2')
    list_hex_area_sqkm.append(round(akm,3))
    list_hex_area_sqm.append(round(am,3))

    
df_meta = pd.DataFrame({"edge_length_km" : list_hex_edge_km,
                        "perimeter_km" : list_hex_perimeter_km,
                        "area_sqkm": list_hex_area_sqkm,
                        "edge_length_m" : list_hex_edge_m,
                        "perimeter_m" : list_hex_perimeter_m,
                        "area_sqm" : list_hex_area_sqm
                       })
                      
df_meta[["edge_length_km","perimeter_km","area_sqkm", "edge_length_m", "perimeter_m" ,"area_sqm"]]



lat_centr_point = selected_region['lat_center'] # -122.382202
lon_centr_point = selected_region['lng_center'] # 37.855068
list_hex_res = []
list_hex_res_geom = []
list_res = range(0,max_res+1)

for resolution in range(0,max_res + 1):
    #index the point in the H3 hexagon of given index resolution
    h = h3.geo_to_h3(lat=lat_centr_point,lng=lon_centr_point, res=resolution)
    list_hex_res.append(h)
    #get the geometry of the hexagon and convert to geojson
    h_geom = { "type" : "Polygon",
               "coordinates": 
                    [h3.h3_to_geo_boundary(h3_address=h,geo_json=True)]
              }
    list_hex_res_geom.append(h_geom)

    
df_resolution_example = pd.DataFrame({"res" : list_res,
                                      "hex_id" : list_hex_res,
                                      "geometry": list_hex_res_geom 
                                     })
df_resolution_example["hex_id_binary"] = df_resolution_example["hex_id"].apply(lambda x: bin(int(x,16))[2:])

pd.set_option('display.max_colwidth',63)
df_resolution_example.head()


HEX_RESOLUTIONS = [7, 8, 9]

dataset_with_hex = [] # 2d array: quadrant, hex resolutions

def assign_hex(df, resolution):
    # assigns a hex_id to row, based on lat/lng and resolution
    
    df["hex_id"] = df.apply(
        lambda row: h3.geo_to_h3(
            row["lat"], row["lng"], resolution),
        axis=1)
    return df

for d in quadrant_dataset:
    interim_dataset = []
    for h in HEX_RESOLUTIONS:
        d = assign_hex(df=d, resolution=h)
        interim_dataset.append(d)
    dataset_with_hex.append(interim_dataset)
    
# index data spatially by h3
def counts_by_hexagon(df, resolution):
    
    '''Use h3.geo_to_h3 to index each data point into the spatial index of the specified resolution.
      Use h3.h3_to_geo_boundary to obtain the geometries of these hexagons'''

    df = df[["lat","lng"]]
    
    df["hex_id"] = df.apply(
        lambda row: h3.geo_to_h3(
            row["lat"], row["lng"], resolution),
        axis = 1) # assign hex_id to row
    
    df_aggreg = df.groupby(by = "hex_id").size().reset_index()
    df_aggreg.columns = ["hex_id", "value"]
    
    df_aggreg["geometry"] =  df_aggreg.hex_id.apply(
        lambda x: 
           {    "type" : "Polygon",
                 "coordinates": 
                [h3.h3_to_geo_boundary(h3_address=x,geo_json=True)]
            }
        )
    
    return df_aggreg


# vis with choropleth map
def hexagons_dataframe_to_geojson(df_hex, file_output = None):
    
    '''Produce the GeoJSON for a dataframe that has a geometry column in geojson format already, along with the columns hex_id and value '''
    
    list_features = []
    
    for i,row in df_hex.iterrows():
        feature = Feature(geometry = row["geometry"] , id=row["hex_id"], properties = {"value" : row["value"]})
        list_features.append(feature)
        
    feat_collection = FeatureCollection(list_features)
    
    geojson_result = json.dumps(feat_collection)
    
    #optionally write to file
    if file_output is not None:
        with open(file_output,"w") as f:
            json.dump(feat_collection,f)
    
    return geojson_result


import branca.colormap as cm
from folium import Map, Marker, GeoJson
from geojson.feature import *
import json

def choropleth_map(df_aggreg, border_color = 'black', fill_opacity = 0.7, initial_map = None, with_legend = False,
                   kind = "linear"):
    #colormap
    min_value = df_aggreg["value"].min()
    max_value = df_aggreg["value"].max()
    m = round ((min_value + max_value ) / 2 , 0)
    
    #take resolution from the first row
    res = h3.h3_get_resolution(df_aggreg.loc[0,'hex_id'])
    
    if initial_map is None:
        initial_map = Map(location= [lat_centr_point, lon_centr_point], zoom_start=11, tiles="cartodbpositron", 
                attr= '© <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="http://cartodb.com/attributions#basemaps">CartoDB</a>' 
            )
        

    #the colormap 
    #color names accepted https://github.com/python-visualization/branca/blob/master/branca/_cnames.json
    if kind == "linear":
        custom_cm = cm.LinearColormap(['green','yellow','red'], vmin=min_value, vmax=max_value)
    elif kind == "outlier":
        #for outliers, values would be -11,0,1
        custom_cm = cm.LinearColormap(['blue','white','red'], vmin=min_value, vmax=max_value)
    elif kind == "filled_nulls":
        custom_cm = cm.LinearColormap(['sienna','green','yellow','red'], 
                                      index=[0,min_value,m,max_value],vmin=min_value,vmax=max_value)
   

    #create geojson data from dataframe
    geojson_data = hexagons_dataframe_to_geojson(df_hex = df_aggreg)
    
    #plot on map
    name_layer = "Choropleth " + str(res)
    if kind != "linear":
        name_layer = name_layer + kind
        
    GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': custom_cm(feature['properties']['value']),
            'color': border_color,
            'weight': 1,
            'fillOpacity': fill_opacity 
        }, 
        name = name_layer
    ).add_to(initial_map)

    #add legend (not recommended if multiple layers)
    if with_legend == True:
        custom_cm.add_to(initial_map)
    
    return initial_map


# test that the hexes are in quadrant that they were assigned
# take datapoints in the top-left quadrant (quadrant 2)

import folium

sanity_hex_map = draw_quadrants_map()
quad_2_res_8_datapoints = dataset_with_hex[1][0]

aggreg = counts_by_hexagon(df=quad_2_res_8_datapoints, resolution=8)  # get counts at x aggreg
sanity_hex_map = choropleth_map(
        df_aggreg=aggreg, initial_map=sanity_hex_map,
        with_legend=False) # place new layer on maps
sanity_hex_map.save('sanity_hex_map.html')
sanity_hex_map


quad_2_res_8_datapoints.head()


import folium

sanity_points_in_hex_map = draw_quadrants_map()
quad_2_res_8_datapoints = dataset_with_hex[1][0]

aggreg = counts_by_hexagon(df=quad_2_res_8_datapoints, resolution=8)  # get counts at x aggreg
sanity_points_in_hex_map = choropleth_map(
        df_aggreg=aggreg, initial_map=sanity_points_in_hex_map,
        with_legend=False) # place new layer on maps

locations = quad_2_res_8_datapoints[['lat', 'lng']]
locationlist = locations.values.tolist()

for point in range(0, len(locationlist)):
    folium.Marker(
        locationlist[point],
        popup=str(locationlist[point])).add_to(sanity_points_in_hex_map)

sanity_points_in_hex_map.save('sanity_points_in_hexes.html')
sanity_points_in_hex_map


import folium

res_map_0_6 = draw_quadrants_map()

for x in range(0, 7):
    print(f'Processing {x} resolution')
    aggreg = counts_by_hexagon(df=raw_rental_df, resolution=x)  # get counts at x aggreg
    res_map_0_6 = choropleth_map(
            df_aggreg=aggreg, initial_map=res_map_0_6,
            with_legend=False) # place new layer on maps
    
folium.map.LayerControl('bottomright', collapsed=False).add_to(res_map_0_6)
res_map_0_6.save('choropleth_0_6.html')
res_map_0_6


import folium

res_map_7_10 = draw_quadrants_map()

for x in range(7, 11):
    print(f'Processing {x} resolution')
    aggreg = counts_by_hexagon(df=raw_rental_df, resolution=x)  # get counts at x aggreg
    res_map_7_10 = choropleth_map(
            df_aggreg=aggreg, initial_map=res_map_7_10,
            with_legend=False) # place new layer on maps
    
folium.map.LayerControl('bottomright', collapsed=False).add_to(res_map_7_10)
res_map_7_10.save('choropleth_7_10.html')
res_map_7_10


import datetime

# get the max time
# subtract timescale t from it, save to a list
TIMESCALES = [30, 60, 90, 180, 9999] # days - last one is "all"
TIMESCALES = [30]
cutoff_dates = [(raw_rental_df.start_date.max() - datetime.timedelta(x)) for x in TIMESCALES]

dataset_with_timescale = [] # 3d array: quadrant, hex, timescale

# for each variant generated so far, filter the specific timescale
for df in quadrant_dataset: # datasets in particular quadrants
    interim_quad_dataset = []
    for date in cutoff_dates:
        interim_quad_dataset.append(df[df['start_date'] >= date])
    dataset_with_timescale.append(interim_quad_dataset)
    
    
    
# walk through 3d array all the way down
# split via jenks natural breaks - collect min, max, mean, median, and 3 random sets
# group by date, save to csv
# groupy by hour, save to csv

import warnings
warnings.filterwarnings('ignore')


def transform_hex_dataset(df_data, timeseries):
    coords_col = ['hex_id']
    grouping = ['hex_id']
    grouping.extend(timeseries)
    
    # create a pivot table
    # index: timeseries
    # columns: hex_id and timeseries    
    hexgrouped_df = df_data.groupby(grouping).size().to_frame().reset_index()
    timeindexed_hexgrouped_df = pd.pivot_table(
        hexgrouped_df,
        values=0,
        index=timeseries,
        columns=coords_col)
    timeindexed_hexgrouped_df.fillna(0, inplace=True)
    
    return timeindexed_hexgrouped_df


def collect_sample_hex_dataset(df_data, df_counts, breaks, timeseries):
    # how to group - by hex_id and the time series we have (daily/hourly)
    timeindexed_hexgrouped_df = transform_hex_dataset(df_data, timeseries)

    out_df_list = []
    
    # iterate through the break groups
    for i in range(len(breaks) - 1):
        hexes = df_counts[(df_counts.value > breaks[i]) & (df_counts.value <= breaks[i+1])].dropna() # get groups
        hexes = hexes.reset_index().sort_values('value')

        out_df = pd.DataFrame(index=list(range(len(timeindexed_hexgrouped_df))),
                              data={'timeseries': timeindexed_hexgrouped_df.index.to_numpy()})
        
        # if fewer than 6 hexes in the break, just add all hexes to the df and output
#         if len(hexes) < 6:
        out_df = out_df.join(
            timeindexed_hexgrouped_df[hexes.hex_id].reset_index().drop(timeseries, axis=1))
        out_df = out_df.set_index('timeseries')
        out_df_list.append(out_df)
#         else:            
#             median_hex_id = hexes.iloc[int(len(hexes)/2)].hex_id  # median
#             max_hex_id = hexes.iloc[int(len(hexes))-1].hex_id # max
#             min_hex_id = hexes.iloc[0].hex_id # min

#             out_df = out_df.join(
#                 timeindexed_hexgrouped_df[median_hex_id].to_frame().reset_index().drop(timeseries, axis=1))
#             out_df = out_df.join(
#                 timeindexed_hexgrouped_df[max_hex_id].to_frame().reset_index().drop(timeseries, axis=1))
#             out_df = out_df.join(
#                 timeindexed_hexgrouped_df[min_hex_id].to_frame().reset_index().drop(timeseries, axis=1))

#             out_df = out_df.set_index('timeseries')
#             out_df_list.append(out_df)
    return out_df_list


# RESOLUTIONS = list(range(max_res + 1))
RESOLUTIONS = [7, 8, 9]
parent_dir = 'known_trips_breakdown'

in_mem_breakdown = {}

import os
try:
    os.mkdir(parent_dir)
except FileExistsError:
    pass


alpha=2
for i, timescales in enumerate(dataset_with_timescale): # df list: quadrants, timescales (then individual dfs)
    # make dir with quadrant label
    quadrant_subdir = f'{parent_dir}/quadrant_{i}'
    quad_label = f'quadrant_{i}'
    in_mem_breakdown[quad_label] = {}
    
    try:
        os.mkdir(quadrant_subdir)
    except FileExistsError:
        pass
    
    for j, df in enumerate(timescales): # df with diff timescales
        # make dir with time scale
        timescale_subdir = f'{quadrant_subdir}/timescale_{TIMESCALES[j]}'
        timescale_label = f'timescale_{TIMESCALES[j]}'
        in_mem_breakdown[quad_label][timescale_label] = {}
        
        try:
            os.mkdir(timescale_subdir)
        except FileExistsError:
            pass
        
        for res in RESOLUTIONS:
            print(f'quadrant: {i}, timescale: {TIMESCALES[j]}, resolution: {res}')
            
            resolution_label = f'res_{res}'
            in_mem_breakdown[quad_label][timescale_label][resolution_label] = {}
            
            # hexbin the points based on resolution
            #   get the counts
            hexbinned_counts = counts_by_hexagon(df, res)
            hexbinned_df = assign_hex(df, res)
            
            # if fewer than 5 hexes in dataset, just log values
            if alpha==2:  # 5 break groups min

                # daily aggr
                timeindexed_hexgrouped_df = transform_hex_dataset(hexbinned_df, ['start_date'])
                out_df = pd.DataFrame(index=list(range(len(timeindexed_hexgrouped_df))),
                                      data={'timeseries': timeindexed_hexgrouped_df.index.to_numpy()})
                for k, hexes in enumerate(timeindexed_hexgrouped_df.columns):
                    hex_count_label = f'quantile_{k}'
                    out_df = out_df.join(
                        timeindexed_hexgrouped_df[hexes].reset_index().drop(['start_date'], axis=1))
                out_df = out_df.set_index('timeseries')
                # change the columns to be lat_lng
                out_df.columns = [str(h3.h3_to_geo(x)) for x in out_df.columns]
                in_mem_breakdown[quad_label][timescale_label][resolution_label]['daily'] = out_df

                filename = f'hex_edge_{df_meta.loc[res].edge_length_m}m_all_hexes_daily.csv'
                # change the columns to be lat_lng
                out_df.to_csv(f'{timescale_subdir}/{filename}')
                
            
                # hourly aggr
                timeindexed_hexgrouped_df = transform_hex_dataset(hexbinned_df, ['start_date', 'start_datetime_hour'])
                out_df = pd.DataFrame(index=list(range(len(timeindexed_hexgrouped_df))),
                                      data={'timeseries': timeindexed_hexgrouped_df.index.to_numpy()})
                for k, hexes in enumerate(timeindexed_hexgrouped_df.columns):
                    hex_count_label = f'quantile_{k}'
                    out_df = out_df.join(
                        timeindexed_hexgrouped_df[hexes].reset_index().drop(['start_date', 'start_datetime_hour'], axis=1))
                out_df = out_df.set_index('timeseries')
                out_df.columns = [str(h3.h3_to_geo(x)) for x in out_df.columns]
                in_mem_breakdown[quad_label][timescale_label][resolution_label]['hourly'] = out_df
                
                filename = f'hex_edge_{df_meta.loc[res].edge_length_m}m_all_hexes_hourly.csv'
                # change the columns to be lat_lng
                out_df.to_csv(f'{timescale_subdir}/{filename}')
                        
           
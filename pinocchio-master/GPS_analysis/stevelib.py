import json
import math

import requests

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
import gpxpy
import numpy as np
import pandas as pd
from pyproj import Proj, transform

from secrets import google_maps_api_key


# read in a gpx file and return a dataframe with a datetime (UTC?), and lat and long columns
def convert_gpx_to_df(filename):
    gpx_file = open(filename, 'r')
    gpx = gpxpy.parse(gpx_file)

    temp_list = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
#                 print('Point at ({0},{1}) {2}'.format(point.latitude, point.longitude, point.time))
                temp_list.append([point.time, point.latitude, point.longitude])
    #             print(dir(point))
    df = pd.DataFrame(temp_list, columns = ['datetime' , 'lat', 'long'])
    return df.set_index('datetime').sort_index()


''' sample sumo query for convert_sumo_to_df()
"'request_method': 'POST" "event_type': 'OTAKEYS_VEHICLE_SYNTHESIS"
| parse "body': '*', 'request_method" as body
| json field=body "synthesis.isEngineRunning"
| json field=body "operationCode"
| json field=body "operationState"
//| json field=body "synthesis.gpsInformation.captureDate" as captureDate
| json field=body "synthesis.gpsInformation.latitude" as lat
| json field=body "synthesis.gpsInformation.longitude" as long
//| json field=body "vehicleSynthesis.gpsInformation.latitude" as vs_lat
//| json field=body "vehicleSynthesis.gpsInformation.longitude" as vs_long
| json field=body "vehicle.extId" as id
| fields -_raw, body
'''

# read in syntheses from sumo logic
# expected to have a column named lat and long
# creates a df with datetime, lat, and long
def convert_sumo_to_df(filename):
    syntheses = pd.read_csv(filename)
    syntheses['datetime'] = pd.to_datetime(syntheses['_messagetime'])
    syntheses['datetime'] = syntheses['datetime']#.dt.tz_localize('UTC').dt.tz_convert(TIMEZONE)
    # syntheses = df.drop_duplicates(subset=['lat', 'long'])  # remove duplicates, keeping the first
    syntheses['lat'] = syntheses['lat'].astype(float)
    syntheses['long'] = syntheses['long'].astype(float)
    syntheses.drop(columns=['_messagetime', '_messagetimems'], inplace=True)
#     syntheses['error'] = np.nan
    return syntheses.set_index('datetime').sort_index()


''' sample SQL query to get app logs for convert_app_logs_to_df()
select distinct * 
from
(
    select 
        from_iso8601_timestamp(ua.timestamp) as _messagetime, 
--         cast(sc.user_id as integer) as customer_id,
        "data-user_location-lat" as lat,
        "data-user_location-lng" as long
  from 
        "data_lake_us_prod"."ma_user_activity" ua
    left join data_lake_us_prod.ma_session_creation sc on ua.session_id = sc.session_id
    where
--         ua.name = 'APP-LAUNCH' 
        ua."tenant_id" = 'daytona-prod'
--         and sc.user_id = '149'
        and [from_iso8601_timestamp("ua"."timestamp")=daterange]
        and "data-user_location-lat" is not null
) a
order by
        1
'''
# read in app logs (containing lat/lngs) from Periscope?
# this data is from SQL/Periscope/Athena
# expects logs with lat, and long, and datetime columns
def convert_app_logs_to_df(filename):
    applogs = pd.read_csv(filename)
    applogs['datetime'] = pd.to_datetime(applogs['datetime'])
    applogs['lat'] = applogs['lat'].astype(float)
    applogs['long'] = applogs['long'].astype(float)
    applogs['datetime'] = applogs['datetime']#.dt.tz_localize('UTC').dt.tz_convert(TIMEZONE)
    # applogs.drop(columns=['_messagetime'], inplace=True)
    # applogs['error'] = np.nan
    return applogs.set_index('datetime').sort_index()


# reads in logs exported from the MENSA Project admin interface
# the id (vin) must be the supplied since mensa doesn't include it in the logs
def convert_mensa_to_df(filename, id, timezone):
    df = pd.read_csv(filename, sep=';')
    df['datetime'] = pd.to_datetime(df['Timestamp']).dt.tz_localize(timezone)
    df['lat'] = df['Latitude'].astype(float)
    df['long'] = df['Longitude'].astype(float)
    df['id'] = id
    df.drop(columns=['Timestamp', 'Latitude', 'Longitude', 'Address', 'BLE Key', 'NFC id'], inplace=True)
    return df.set_index('datetime').sort_index()


# read in the json from datapipeline s3 bucket files
def convert_data_pipeline_to_df(filename):
    result = []
    f = open(filename)
    for line in f:
        data = json.loads(line)
        if data['name'] == 'APP-LOCATION-UPDATE':
            result.append([data['timestamp'], data['user_id'], data['data']['user_location']['lat'], data['data']['user_location']['lng']])
    df = pd.DataFrame(result, columns = ['datetime', 'id', 'lat', 'long'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['id'] = df['id'].astype(str)
    df['lat'] = df['lat'].astype(float)
    df['long'] = df['long'].astype(float)
    return df.set_index('datetime').sort_index()


# takes in a list of long and a list of lat values, and converts them from WGS84 to Web Mercator 
def transform_wgs84_to_web_mercator(long, lat):
    y2, x2 = transform(
        Proj(init='epsg:4326'), 
        Proj(init='epsg:3857'), 
        long, # longitude first, latitude second.
        lat
    )  
    return y2, x2


# takes in figure, a list of latitudes, a list of longitudes, and a variable number of plot parameters (like color) and plots all the points on the figure
def plot2(figure, lat_values, long_values, **kwargs):
    x2, y2 = transform_wgs84_to_web_mercator(long_values, lat_values) 
    # for some reason this ColumnDataSource doesn't work with single value
    source = ColumnDataSource({'x': x2, 'y': y2})
    figure.circle(x='x', y='y', source=source, **kwargs)
#     return figure


# takes in a [time], lat, long df, and returns a snapped to roads lat, long df
def snap_to_roads(df):
    # convert the lat/longs into a string
    # https://roads.googleapis.com/v1/snapToRoads?path=-35.27801,149.12958|-35.28032,149.12907|-35.28099,149.12929|-35.28144,149.12984|-35.28194,149.13003|-35.28282,149.12956|-35.28302,149.12881|-35.28473,149.12836&interpolate=true&key=YOUR_API_KEY
    string = ''
    for index, row in df.iterrows():  # surely a better way to do this, a more pandas way...
#         print(row)
        string += f'{row.lat},{row.long}|'
#     print(string)
    key = google_maps_api_key  # do not make public!!!!
    url = 'https://roads.googleapis.com/v1/snapToRoads'
    params = {
        'path': string[:-1],
        'key': key,
        'interpolate': 'False'
    }
    
    r = requests.get(url, params=params)

    # parse output
    temp = []
    for row in r.json()['snappedPoints']:
    #     print(row)
        temp.append([row['location']['latitude'], row['location']['longitude']])
    df2 = pd.DataFrame(temp, columns = ['lat', 'long'])
    return df2


# use geopy.distance instead?
# from https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
# takes in a list of longitudes and latitudes, and another list of the same, and calculates the meters between
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    meters = 6367e3 * c
    return meters


# make a circle in bokeh because it doesn't do circles accurately on a map
def make_circle(figure, centerLat, centerLon, radius, **kwargs):
    # m the following code is an approximation that stays reasonably accurate for distances < 100km
    # parameters
    N = 1*360 # number of discrete sample points to be generated along the circle

    # generate points
    lats = []
    longs = []
    for k in range(N):
        # compute
        angle = math.pi*2*k/N
        dx = radius*math.cos(angle)
        dy = radius*math.sin(angle)
        lats.append(centerLat + (180/math.pi)*(dy/6378137))
        longs.append(centerLon + (180/math.pi)*(dx/6378137)/math.cos(centerLat*math.pi/180))

    x0, y0 = transform_wgs84_to_web_mercator(longs, lats)
    figure.line(x=x0, y=y0, **kwargs)

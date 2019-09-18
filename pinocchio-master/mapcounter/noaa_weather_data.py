import requests
from datetime import datetime, timezone
from pyproj import Proj
import geopandas as gpd
from bokeh.layouts import row
from bokeh.models import LinearColorMapper, ColorBar, BasicTicker, HoverTool
from bokeh.palettes import Magma8 as palette
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.util.hex import cartesian_to_axial, hexbin
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.transform import linear_cmap
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.interpolate import griddata


# build sdk later?
class NOAA_Interface(object):

    def __init__(self, token):
        self.url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'
        self.token_header = {'token': token}

    def get(self, req_type, params):
        req = requests.get(self.url + req_type,
                           headers=self.token_header, params=params)

        if req.status_code != requests.codes.ok:
            print('Error: {}'.format(req.status_code))
            print(req.json().get('status'))
            print(req.json().get('message'))
        else:
            return req.json()

    def stations(self, **kwargs):
        '''
        list: https://www.ncdc.noaa.gov/cdo-web/api/v2/stations
        '''
        return self.get('stations', kwargs)

    def data(self, datasetid, startdate, enddate, **kwargs):
        '''
        required:
            datasetid: https://www.ncdc.noaa.gov/cdo-web/api/v2/datasets
            startdate: yyyy-mm-dd
            enddate: yyyy-mm-dd
        good optionals: locationid, stationid, units, datatypeid
        '''
        required_params = {'datasetid': datasetid, 'startdate': startdate,
                           'enddate': enddate}
        try:
            combined_params = dict(**required_params, **kwargs)
        except TypeError:
            print('Required key detected in kwargs!')
            return None

        return self.get('data', combined_params)


def filter_by_date(df, startdate, enddate):

    df['_time'] = pd.to_datetime(df['_time'])  # convert the time column to an actual time column
    start = datetime.fromisoformat(startdate).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(enddate).replace(tzinfo=timezone.utc)

    return df[(df['_time'] > start) & (df['_time'] <= end)]


def convert_to_mercator(lngs, lats):
    projection = Proj(init='epsg:3857')
    xs = []
    ys = []
    for lng, lat in zip(lngs, lats):
        x, y = projection(lng, lat)
        xs.append(x)
        ys.append(y)

    return xs, ys


def calculate_hex_precipitation_scores(_hexes, stations_df):
    # drop all stations that have no values to report or no distance
    no_na_stations_df = stations_df.dropna()
    print(_hexes)
    # points without data
    _hexes_values = griddata(
            (no_na_stations_df['q'], no_na_stations_df['r']),
            no_na_stations_df['value'],
            (_hexes['q'], _hexes['r']),
            fill_value=0.0,
            method='linear')
    return _hexes_values


def main():
    # appopens
    DATAFILENAME = 'appopens_20190115_20190214.csv'

    # get locations
    # BOUNDING_BOX = '37.638788,-122.665213,37.942281,-122.184490'
    BOUNDING_BOX = '37.097250,-122.487705,38.127344,-121.798901'
    # get start and end times
    STARTDATE = '2019-01-15'
    ENDDATE = '2019-02-14'
    DATEOFINTEREST = datetime(2019, 2, 5).isoformat()
    # dataset
    DATASETID = 'GHCND' # general weather data: Global Historical Climatology Network
    # get desired data types
    DATATYPE = 'PRCP' # precipitation

    TOKEN = 'BOkSxvxsxBgfLWcyAMYIztxImLYWjZXu'

    noaa_interface = NOAA_Interface(TOKEN)

    # get stations
    station_req_params = {'limit': 50, 'startdate': STARTDATE,
                          'enddate': ENDDATE, 'extent': BOUNDING_BOX}
    stations = noaa_interface.stations(**station_req_params).get('results')

    data_req_params = {'units': 'metric', 'datatypeid': DATATYPE,
                       'limit': 1000, 'sortfield': 'date', 'sortorder': 'asc',
                       'stationid': [s.get('id') for s in stations]}
    data = noaa_interface.data(
            DATASETID, STARTDATE, ENDDATE, **data_req_params).get('results')

    # {date1: {station1: value1, station2: value2}}
    precip_data_per_day = {}
    for res in data:
        if res.get('date') in precip_data_per_day and res.get('value') > 0:
            precip_data_per_day[res.get('date')][res.get('station')] = res.get('value')
        else:
            precip_data_per_day[res.get('date')] = {res.get('station'): res.get('value')}

    # put stations on map
    map_figure = figure(
        x_range=(-13618976.4221, -13605638.1607),  # bounding box for starting view
        y_range=(4549035.0828, 4564284.2700),
        x_axis_type='mercator',
        y_axis_type='mercator',
        title='Appopens to Rainfall for {}'.format(DATEOFINTEREST),
        tooltips=[('count', '@counts'),
                  ('(q, r)', '(@q, @r)'),
                  ('Rain Coefficient', '@rc'),
                  ('Fill Color Hex Code', '@fc')]
    )
    map_figure.add_tile(CARTODBPOSITRON)

    # appopen data points
    # read in the data points
    df = pd.read_csv(DATAFILENAME)
    filtered_df = filter_by_date(df, STARTDATE, ENDDATE)

    # hex binning
    df_x, df_y = convert_to_mercator(df['lng'].to_list(), df['lat'].to_list())
    bins = hexbin(x=np.asarray(df_x), y=np.asarray(df_y), size=500)

    # get points on dateofinterest
    projection = Proj(init='epsg:3857')
    station_lng = []
    station_lat = []
    for s in stations:
        x, y = projection(s.get('longitude'), s.get('latitude'))
        station_lng.append(x)
        station_lat.append(y)

    # draw stations
    map_figure.circle(x=station_lng, y=station_lat, size=15, fill_color='blue', fill_alpha=0.8)

    # get precipitation data for the given day
    # convert station coordinates to hex
    # in order to get the weighted sum for precipitation in a given hex
    station_names = [s.get('name') for s in stations]
    station_ids = [s.get('id') for s in stations]
    station_q, station_r = cartesian_to_axial(
            np.asarray([station_lng]),
            np.asarray([station_lat]),
            size=500,
            orientation='pointytop')
    station_values = []
    for _id in station_ids:
        station_values.append(
                precip_data_per_day.get(DATEOFINTEREST).get(_id))
    station_data = {'q': station_q.flatten(), 'r': station_r.flatten(),
                    'name': station_names, 'value': station_values,
                    'id': station_ids}

    # calculate precipitation scores for each hex
    station_df = pd.DataFrame(data=station_data)
    _hex_precip_scores = calculate_hex_precipitation_scores(bins, station_df)
    bins.insert(len(bins.columns), 'rc', np.asarray(_hex_precip_scores))

    r_max = bins['counts'].max()
    g_max = bins['rc'].max()

    color_plot = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(bins['counts']/r_max*256, bins['rc']/g_max*256)]
    bins.insert(len(bins.columns), 'fc', np.asarray(color_plot))
    # map_figure.hex_tile(q='q', r='r', size=500, source=bins, hover_color="pink", hover_alpha=0.8, fill_alpha=0.8, fill_color='fc')
    map_figure.hex_tile(q='q', r='r', size=500, source=bins, hover_color="pink", hover_alpha=0.8, fill_alpha=0.8,
                        fill_color='fc')
    print(bins.loc[bins['rc'] > 0])

    output_file("stations.html")
    show(map_figure)


if __name__ == '__main__':
    main()

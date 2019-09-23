import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# from workbook "read in data II"


'''
Takes in a GeoJSON regions file and a CSV (time, lat, long) Splunk export 

Returns a dataframe with regions and a dataframe with the Splunk data bucketed
'''
def getData(GeoJSON_filename, Splunk_export_filename):
    regions = gpd.read_file(GeoJSON_filename)

    # Splunk_export_filename = '1549304575_89608.csv'
    df = pd.read_csv(Splunk_export_filename)
    df['datetime'] = pd.to_datetime(df['_time'])
    df['coordinates'] = list(zip(df.lng, df.lat))  # create a coordinates pair
    df['geometry'] = df['coordinates'].apply(Point)  # create POINT objects
    df.drop(['_time', 'lat', 'lng', 'coordinates'], axis=1, inplace=True)  # get rid of extra crap

    points = gpd.GeoDataFrame(df, geometry='geometry')

    points_and_regions = gpd.sjoin(regions, points, how="inner", op='contains')
    points_and_regions.drop(['geometry'], axis=1, inplace=True)  # get rid of extra crap
    points_and_regions.set_index(['datetime'], inplace=True)
    points_and_regions.index = points_and_regions.index.tz_localize('UTC').tz_convert('America/Los_Angeles')


    lol = points_and_regions.groupby('id').resample('3H').count()
    t_index = pd.DatetimeIndex(start=points_and_regions.index.min().date(), end=points_and_regions.index.max().date()  + pd.DateOffset(1), freq='3H', tz='America/Los_Angeles')[:-1] # don't include the new day that is the last point >:O
    lol=lol.unstack(level=0).fillna(0).reindex(t_index).stack('id').swaplevel(0,1).sort_index()
    lol.rename({'id': 'count'}, axis='columns', inplace=True)

    lol.reset_index(inplace=True)
    lol.rename({'level_1': 'datetime'}, axis='columns', inplace=True)  # somehow datetime loses its name in the unstacking mess 

    lol['day_of_week'] = lol.datetime.dt.dayofweek
    lol['hour'] = lol.datetime.dt.hour
    lol['hour_bucket'] = lol.groupby([lol.id, lol.datetime.dt.date]).cumcount()

    lol.set_index(['id', 'day_of_week', 'hour_bucket'], inplace=True)

    return regions, lol


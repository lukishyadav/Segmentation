from bokeh.layouts import row
from bokeh.models import LinearColorMapper, ColorBar, BasicTicker
from bokeh.palettes import Magma8 as palette
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.tile_providers import CARTODBPOSITRON
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


BUCKET_HOURS = 3  # how many hours make up a single bucket of time 
START_TIME = 13  # clock hour number (0-23)
END_TIME = START_TIME + BUCKET_HOURS
DAY_OF_WEEK = 'Friday'
REGION_OF_INTEREST = 0  # index of the region to start looking in detail at


region_filename = 'berkeley-s7996m-geojson.json'  # file with geojson regions
data_filename = '1549304575_89608.csv'  # file with time, lat, long data


def convert_GeoPandas_to_Bokeh_format(gdf):
    """
    Function to convert a GeoPandas GeoDataFrame to a Bokeh
    ColumnDataSource object.
    
    :param: (GeoDataFrame) gdf: GeoPandas GeoDataFrame with polygon(s) under
                                the column name 'geometry.'
                                
    :return: ColumnDataSource for Bokeh.
    """
    gdf_new = gdf.drop('geometry', axis=1).copy()
    gdf_new['x'] = gdf.apply(getGeometryCoords, 
                             geom='geometry', 
                             coord_type='x', 
                             shape_type='polygon', 
                             axis=1)
    
    gdf_new['y'] = gdf.apply(getGeometryCoords, 
                             geom='geometry', 
                             coord_type='y', 
                             shape_type='polygon', 
                             axis=1)
    
    return ColumnDataSource(gdf_new)

def getGeometryCoords(row, geom, coord_type, shape_type):
    """
    Returns the coordinates ('x' or 'y') of edges of a Polygon exterior.
    
    :param: (GeoPandas Series) row : The row of each of the GeoPandas DataFrame.
    :param: (str) geom : The column name.
    :param: (str) coord_type : Whether it's 'x' or 'y' coordinate.
    :param: (str) shape_type
    """
    
    # Parse the exterior of the coordinate
    if shape_type == 'polygon':
        exterior = row[geom].geoms[0].exterior
        if coord_type == 'x':
            # Get the x coordinates of the exterior
            return list( exterior.coords.xy[0] )    
        elif coord_type == 'y':
            # Get the y coordinates of the exterior
            return list( exterior.coords.xy[1] )
    elif shape_type == 'point':
        exterior = row[geom]
        if coord_type == 'x':
            # Get the x coordinates of the exterior
            return  exterior.coords.xy[0][0] 
        elif coord_type == 'y':
            # Get the y coordinates of the exterior
            return  exterior.coords.xy[1][0]

# returns the data_df points that are within geometry 
def computeHourTimeBuckets(geometry, data_gdf, data_df):
    print('START')
    within_mask = data_gdf.within(geometry)  # mask array of just the data points within this geometry 
    print('END')
    return data_df[within_mask].groupby(pd.Grouper(freq=str(BUCKET_HOURS) + 'H')).count()


# 
def filterByTimeAndDayOfWeek(buckets, time_start, time_end, day_of_the_week):
    buckets2 = buckets.between_time(time_start, time_end, include_end=False)  # filter by time
    b = buckets2[(buckets2.index.weekday_name == day_of_the_week)]  # filter by day of week

    x_values = b.index.view('int64') # unix time values because we can't do linear regression on actual datetimes
    y_values = b['lat'].values  # bucket counts

    regr = linear_model.LinearRegression()
    # fit on all the data except the last point
    regr.fit(x_values[:-1].reshape(-1, 1), y_values[:-1])
    # compute what the linear model would show on all points
    y_linear = regr.predict(x_values.reshape(-1, 1))
    # we return just the last point separately for ease of plotting
    return b.index.values, y_values, y_linear[-1], y_linear


output_file("tile.html")


map_figure = figure(
    x_range=(-13618976.4221, -13605638.1607),  # bounding box for starting view
    y_range=(4549035.0828, 4564284.2700),
    x_axis_type='mercator', 
    y_axis_type='mercator',
    title='Predicted value',
    tooltips=[('Region ID', '@id'),
              ('Name', '@BEAT'), 
              #('Subbeat', '@Subbeat'), 
              ('Predicted value', '@predicted_value')]
)
map_figure.add_tile(CARTODBPOSITRON)

# read in the regions from the berkeley website
regions = gpd.read_file(region_filename)[0:2]  # TODO: remove me
# print(regions)

print(regions['id'])
print(regions.index.values)

# read in the data points
df = pd.read_csv(data_filename)
df['_time'] = pd.to_datetime(df['_time'])  # convert the time column to an actual time column
df = df.set_index('_time') # set the correct time index
# print(df)

# Creating a Geographic data frame 
gdf = gpd.GeoDataFrame(
    df, 
    crs={'init': 'epsg:4326'},
    geometry=[Point(xy) for xy in zip(df['lng'], df['lat'])]
)


# get the buckets
regions['buckets'] = regions['geometry'].apply(lambda geometry: computeHourTimeBuckets(geometry, gdf, df))
# for every region for a specific timeframe, get the x and y data points, as well as the y points from the linear regression, and the final linear regression value (predicted_value)
regions['x_values'], regions['y_values'], regions['predicted_value'], regions['y_linear'] = zip(*regions['buckets'].apply(lambda bucket: filterByTimeAndDayOfWeek(bucket, str(START_TIME) + ':00', str(END_TIME) + ':00', DAY_OF_WEEK)))

print(regions)

patches_data = convert_GeoPandas_to_Bokeh_format(regions.to_crs(epsg=3857).drop('buckets', axis=1))  # convert to web mercator projection, don't include buckets as it's not convertable
color_mapper = LinearColorMapper(palette=palette, low=0, high=max(regions['predicted_value']))
map_figure.patches('x', 
    'y', 
    source=patches_data, 
    fill_color={'field': 'predicted_value', 'transform': color_mapper}, 
    fill_alpha=1.0)
map_figure.add_layout(ColorBar(color_mapper=color_mapper, ticker=BasicTicker()))  # add a legend


# second plot
detailed_figure = figure(
    title='{}s from {} to {} for Region ID {}'.format(DAY_OF_WEEK, str(START_TIME) + ':00', str(END_TIME) + ':00', regions.iloc[REGION_OF_INTEREST].id),
    x_axis_type="datetime"
)

# the actual data points, minus the last one
detailed_figure.scatter(
    regions['x_values'].values[REGION_OF_INTEREST][:-1], 
    regions['y_values'].values[REGION_OF_INTEREST][:-1],
    fill_color='white',
    size=8
)

# the actual last data point
detailed_figure.scatter(
    regions['x_values'].values[REGION_OF_INTEREST][-1], 
    regions['y_values'].values[REGION_OF_INTEREST][-1],
    color='green',
    fill_color='white',
    size=8
)

# the linear regression
detailed_figure.line(
    regions['x_values'].values[REGION_OF_INTEREST], 
    regions['y_linear'].values[REGION_OF_INTEREST],
    color='red',
)

# the predicted last data point
detailed_figure.scatter(
    regions['x_values'].values[REGION_OF_INTEREST][-1], 
    regions['predicted_value'].values[REGION_OF_INTEREST],  # could also use y_linear[-1]
    color='red',
    fill_color='white',
    size=8
)

show(row(map_figure, detailed_figure))
'''
TODO: 
add day of week selector
add hour bucket selector
add timezones

normalize legend based on active bucket min/max or global min/max

make detailed figure x labels/grid match up with dates
'''

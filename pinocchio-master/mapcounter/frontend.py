from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox
from bokeh.models import LinearColorMapper, ColorBar, BasicTicker
from bokeh.palettes import Plasma11 as palette
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.models.widgets import RadioButtonGroup, Slider, Paragraph, Select
# import geopandas as gpd
# import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

from backend import getData


# # nice way to get the data, returns a list of dates and a list of values
# def getData(region_id=slice(None), day_of_week=0, hour_bucket=0):
#     temp = data.loc[(region_id, day_of_week, hour_bucket), ('count')]
#     return temp.index.get_level_values(0).values, temp.values

def convert_GeoPandas_to_Bokeh_format(gdf):
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
    
    #return ColumnDataSource(gdf_new)
    return gdf_new['x'], gdf_new['y']

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



GeoJSON_filename = 'berkeley-s7996m-geojson.json'  # file with geojson regions
# DOESN'T CURRENTLY WORK BECAUSE ONLY HAS 1 WEEK Splunk_export_filename = '1550170191_321549.csv'  # file with time, lat, long data
Splunk_export_filename = '1549304575_89608.csv'  # file with time, lat, long data


regions, data = getData(GeoJSON_filename, Splunk_export_filename)
regions.sort_values(by=['id'], inplace=True) # to match the order of the ids in the data df
regions_list = data.loc[(slice(None), slice(None), slice(None)), slice(None)].index.get_level_values(0).unique()

p = Paragraph(text="""This counts the number of app opens, per geographic region, per day of the week, per bucke to hours (in this case 3 hour buckets).  The map on the left is a chloropleth map which shows the mean number of app opens.  The chart on the rigth shows the actual app opens for that day of the week and bucket of time, over time.""",)


rx, ry = convert_GeoPandas_to_Bokeh_format(regions.to_crs(epsg=3857))  # convert to web mercator projection, don't include buckets as it's not convertable

map_figure = figure(
    x_range=(-13618976.4221, -13605638.1607),  # bounding box for starting view
    y_range=(4555035.0828, 4570284.2700),
    x_axis_type='mercator', 
    
    
    y_axis_type='mercator',
    # title='Predicted value',
    # tooltips=[('Region ID', '@id'),
    #           ('Name', '@BEAT'), 
    #           #('Subbeat', '@Subbeat'), 
    #           ('Predicted value', '@predicted_value')]
    tooltips=[('Id', '@id'), ('Count', '@count')],
    title='Mean app opens',
)
map_figure.xaxis.axis_label = 'Longitude'
map_figure.yaxis.axis_label = 'Latitude'

color_mapper = LinearColorMapper(palette=palette, low=0, high=data.loc[(slice(None), slice(None), slice(None)), ('count')].max(level=0).max())  # TODO: fix this
map_figure.add_layout(ColorBar(color_mapper=color_mapper, ticker=BasicTicker()))


# making a list of alphas to highlight which region is selected, starting the first one
temp = [0.0 for x in regions_list]
temp[0] = 1.0
map_data = ColumnDataSource(
    data=dict(
        x=rx,
        y=ry,
        count=data.loc[(slice(None), 0, 0), ('count')].mean(level=0).values,
        id=data.loc[(slice(None), 0, 0), ('count')].mean(level=0).index.get_level_values(0).values,
        # predicted_value=regions['predicted_value'],
        # id=regions['id'],
        # name=regions['BEAT'],
        line_alpha=temp,
    )
)

map_figure.patches(
    'x', 
    'y', 
    source=map_data, 
    fill_alpha=0.8,
    fill_color={'field': 'count', 'transform': color_mapper}, 
    line_alpha='line_alpha',
    line_color='yellow',
    line_width=5,

)
map_figure.add_tile(CARTODBPOSITRON)  # the map background


# by_hour_chart = figure(
#     #title='{}s from {} to {} for Region ID {}'.format(DAY_OF_WEEK, str(START_TIME) + ':00', str(END_TIME) + ':00', regions.iloc[REGION_OF_INTEREST].id),
#     # x_axis_type="datetime",
#     # y_range=(0, current_max+2)
#     title='For s7996m.20, count vs. hour',
#     y_range=(0, data.loc[(slice(None), slice(None), slice(None)), ('count')].max(level=0).max()),
# )
# by_hour_chart.xaxis.axis_label = 'Hour (number)'
# by_hour_chart.yaxis.axis_label = 'Count'


# by_hour_chart_data = ColumnDataSource(
#     data=dict(
#         x=data.loc[(regions_list[0], 0, slice(None)), ('hour')].tolist(), 
#         y=data.loc[(regions_list[0], 0, slice(None)), ('count')].tolist(), 
#     )
# )

# # the actual data points
# by_hour_chart.scatter(
#     'x', 
#     'y',
#     source=by_hour_chart_data,
#     fill_color='white',
#     size=8
# )


by_weekday_chart = figure(
    #title='{}s from {} to {} for Region ID {}'.format(DAY_OF_WEEK, str(START_TIME) + ':00', str(END_TIME) + ':00', regions.iloc[REGION_OF_INTEREST].id),
    x_axis_type="datetime",
    # y_range=(0, current_max+2)
    title='Count vs.day of the week (at a specific hour)',
    y_range=(0, data.loc[(slice(None), slice(None), slice(None)), ('count')].max(level=0).max()),
)
by_weekday_chart_data = ColumnDataSource(
    data=dict(
        x=data.loc[(regions_list[0], 0, 0), ('datetime')].tolist(), 
        y=data.loc[(regions_list[0], 0, 0), ('count')].tolist(), 
    )
)

# x = data.loc[(regions_list[0], 0, 0), ('count')].view('int64').values

# y = data.loc[(regions_list[0], 0, 0), ('count')].values

# regr = linear_model.LinearRegression()
# # fit on all the data except the last point
# regr.fit(x[:-1].reshape(-1, 1), y[:-1])
# # compute what the linear model would show on all points
# y_linear = regr.predict(x.reshape(-1, 1))

# print(data.loc[(regions_list[0], 0, 0), ('datetime')].tolist())
# print(x)
# print(y_linear)

# by_weekday_chart_data_linear = ColumnDataSource(
#     data=dict(
#         x2=data.loc[(regions_list[0], 0, 0), ('datetime')].values, 
#         y2=y_linear, 
#     )
# )

# by_weekday_chart_data_all = ColumnDataSource(
#     data=dict(
#         x=data.loc[(regions_list[0], slice(None), 0), ('datetime')].tolist(), 
#         y=data.loc[(regions_list[0], slice(None), 0), ('count')].tolist(), 
#     )
# )

# all the points
# by_weekday_chart.scatter(
#     'x', 
#     'y',
#     source=by_weekday_chart_data_all,
#     fill_color='gray',
#     line_color='gray',
#     line_alpha = 0.5,
#     fill_alpha = 0.5,
#     size=4
# )
# just the selected hour point
by_weekday_chart.scatter(
    'x', 
    'y',
    source=by_weekday_chart_data,
    # fill_color='white',
    size=6
)
# the linear one
# by_weekday_chart.line(
#     'x2', 
#     'y2',
#     source=by_weekday_chart_data_linear,
#     line_color='red',
# )


def update_region(attrname, old, new):
    temp = data.loc[(slice(None), weekday_selector.active, hour_selector.active), ('count')].mean(level=0)
    map_data.data['count'] = temp.values
    map_data.data['id'] = temp.index.get_level_values(0).values
    map_data.data['line_alpha'] = [1.0 if x == new else 0.0 for x in regions_list]

    by_weekday_chart_data.data['x'] = data.loc[(new, weekday_selector.active, hour_selector.active), ('datetime')].tolist()
    by_weekday_chart_data.data['y'] = data.loc[(new, weekday_selector.active, hour_selector.active), ('count')].tolist()


# this gets called when the weekday radio buttons change
def update_weekday(attrname, old, new):
    temp = data.loc[(slice(None), new, hour_selector.active), ('count')].mean(level=0)
    map_data.data['count'] = temp.values
    map_data.data['id'] = temp.index.get_level_values(0).values

    # by_hour_chart_data.data['x'] = data.loc[(region_selector.value, new, slice(None)), ('hour')].tolist()
    # by_hour_chart_data.data['y'] = data.loc[(region_selector.value, new, slice(None)), ('count')].tolist()

    by_weekday_chart_data.data['x'] = data.loc[(region_selector.value, new, hour_selector.active), ('datetime')].tolist()
    by_weekday_chart_data.data['y'] = data.loc[(region_selector.value, new, hour_selector.active), ('count')].tolist()

    # by_weekday_chart_data_all.data['x'] = data.loc[(region_selector.value, new, slice(None)), ('datetime')].tolist()
    # by_weekday_chart_data_all.data['y'] = data.loc[(region_selector.value, new, slice(None)), ('count')].tolist()

def update_hour(attrname, old, new):
    temp = data.loc[(slice(None), weekday_selector.active, new), ('count')].mean(level=0)
    map_data.data['count'] = temp.values
    map_data.data['id'] = temp.index.get_level_values(0).values

    # by_hour_chart_data.data['x'] = data.loc[(region_selector.value, weekday_selector.active, slice(None)), ('hour')].tolist()
    # by_hour_chart_data.data['y'] = data.loc[(region_selector.value, weekday_selector.active, slice(None)), ('count')].tolist()

    by_weekday_chart_data.data['x'] = data.loc[(region_selector.value, weekday_selector.active, new), ('datetime')].tolist()
    by_weekday_chart_data.data['y'] = data.loc[(region_selector.value, weekday_selector.active, new), ('count')].tolist()

    # by_weekday_chart_data_all.data['x'] = data.loc[(region_selector.value, weekday_selector.active, slice(None)), ('datetime')].tolist()
    # by_weekday_chart_data_all.data['y'] = data.loc[(region_selector.value, weekday_selector.active, slice(None)), ('count')].tolist()



region_selector = Select(title="Select region:", value=regions_list[0], options=regions_list.tolist())
region_selector.on_change('value', update_region)


weekday_selector = RadioButtonGroup(
    labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],  # TODO: replace with non-hardcoded values
    active=0
)
weekday_selector.on_change('active', update_weekday)

list_of_hours= [str(x).zfill(2) + ':00' for x in list(sorted(data.loc[(slice(None), slice(None), slice(None)), ('hour')].unique()))]
hour_selector = RadioButtonGroup(
    labels=list_of_hours,
    active=0
)
hour_selector.on_change('active', update_hour)

curdoc().add_root(
    column(
        p, 
        widgetbox(weekday_selector, hour_selector, width=1000),
        widgetbox(region_selector),
        # row(map_figure, by_hour_chart, by_weekday_chart),

        row(map_figure, by_weekday_chart),

    )
)

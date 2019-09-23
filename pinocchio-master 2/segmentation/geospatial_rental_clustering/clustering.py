import pandas as pd
import numpy as np
import calendar
import logging
import sys
from bokeh.layouts import row, column, widgetbox
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.models.widgets import RadioButtonGroup, PreText, Slider
from bokeh.models.ranges import Range1d
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from bokeh.io import curdoc
from bokeh.palettes import Category20
from bokeh.transform import factor_cmap
from pyproj import Proj

"""
TODOs:
    - write in handling UTC time (datalake doesn't allow for adjusting the timezone
    - write in handling isoformat (pandas?)
    - add method to query datalake
      - add option to query
      - add option to save the datafile
"""

# ==== constants ====
kms_per_radian = 6371.0088
distance_in_km = 0.250  # 250 meters
epsilon = distance_in_km/kms_per_radian
palette_range = [str(x) for x in range(0, 20)]
min_n_static = 10
n_percent = 1

# ==== map constants ====
# bounding box for starting view
map_repr = 'mercator'
map_title = 'Cluster Analysis'
x_min = -13618976.4221
x_max = -13605638.1607
y_min = 4549035.0828
y_max = 4564284.2700

# ==== chart constants ====
chart_title = 'Cluster Counts'

# ==== data slicing ====
TIMEZONE = 'America/Los_Angeles'
rental_file = 'successful_rentals_data.csv'
lat_col = 'start_location_lat'
lng_col = 'start_location_lng'
time_col = 'start_datetime'
lat_merc_proj = 'start_location_merc_lat'
lng_merc_proj = 'start_location_merc_lng'

# ==== set up datasource ====
# This allows us to modify the values provided to the figures
# and provide live visual map updates
noise_source = ColumnDataSource()
datapoints_source = ColumnDataSource()
cluster_chart_source = ColumnDataSource()


# ==== functions =====
def convert_to_mercator(lngs, lats):
    projection = Proj(init='epsg:3857')
    xs = []
    ys = []
    for lng, lat in zip(lngs, lats):
        x, y = projection(lng, lat)
        xs.append(x)
        ys.append(y)
    return xs, ys


def set_clusters(df):
    # set the data up for clustering
    min_n = int(df.shape[0]/100) * n_percent  # 1% of samples available
    X = np.column_stack((df[lng_col], df[lat_col]))
    # clustering
    dbscan = DBSCAN(eps=epsilon, min_samples=max(min_n, min_n_static),
                    algorithm='ball_tree',
                    metric='haversine').fit(np.radians(X))

    # create list of clusters and labels
    n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    n_noise_ = list(dbscan.labels_).count(-1)

    # label the points unique to their cluster
    unique_labels = set(dbscan.labels_)
    df['label'] = [str(label) for label in dbscan.labels_]

    return df


def plot_points(map_figure, noise_source, datapoints_source):
    noise_point_size = 1
    cluster_point_size = 10

    # plot points on map
    map_figure.circle(x='x', y='y', size=noise_point_size,
                      fill_alpha=0.2, source=noise_source)

    map_figure.circle(x='x', y='y', size=cluster_point_size,
                      fill_alpha=1, source=datapoints_source,
                      fill_color=factor_cmap(
                          'label',
                          palette=Category20[20],
                          factors=palette_range),
                      line_color='black')


def associate_chart(chart, datasource):
    chart.vbar(x='x', top='counts', source=datasource, width=0.9,
               line_color='black',
               fill_color=factor_cmap(
                   'x', palette=Category20[20], factors=palette_range))


def main():

    def collect_and_group_data(datafile):
        datetime_columns = ['start_datetime', 'end_datetime']

        master_df = pd.read_csv(
                datafile,
                parse_dates=datetime_columns,
                infer_datetime_format=True
                )

        # ==== format the data ====
        for col in datetime_columns:
            master_df[col] = master_df[col].dt.tz_convert('UTC').dt.tz_convert(TIMEZONE)
        master_df = master_df.dropna()
        master_df['hour'] = master_df[time_col].dt.hour
        master_df['dow'] = master_df[time_col].dt.day_name()
        master_df['label'] = np.nan  # initialize with no values

        master_df[lng_merc_proj], master_df[lat_merc_proj] = convert_to_mercator(
            master_df[lng_col], master_df[lat_col])

        # ==== group, apply clustering, regroup for recall ====
        grouped_df = master_df.groupby(['dow', 'hour'])
        grouped_df = grouped_df.apply(set_clusters)
        grouped_df = grouped_df.groupby(['dow', 'hour'])

        return grouped_df

    # === button callback ===
    def update(attr, old, new):

        # get the new groups
        dow = list(calendar.day_name)[weekday_selector.active]
        filtered_df = rental_groupby.get_group((dow, hour_selector.active))
        logging.info(f'Loading DOW: {dow}, Hour: {hour_selector.active}')

        # separate out the noise and the clustered points
        noise_df = filtered_df[filtered_df['label'] == '-1']
        datapoints_df = filtered_df[filtered_df['label'] != '-1']

        # modify columndatasources to modify the figures
        noise_source.data = dict(
            x=noise_df[lng_merc_proj],
            y=noise_df[lat_merc_proj],
            label=noise_df['label']
            )
        datapoints_source.data = dict(
            x=datapoints_df[lng_merc_proj],
            y=datapoints_df[lat_merc_proj],
            label=datapoints_df['label']
            )

        # set the labels
        labels_size = len(set(datapoints_source.data['label']))
        labels_set = [str(x) for x in range(0, labels_size)]
        counts = [len(datapoints_df[
            datapoints_df['label'] == x]) for x in labels_set]
        cluster_chart_source.data = dict(
            x=labels_set,
            counts=counts
            )

        # collect data for stats
        n_clusters = len(datapoints_df['label'].unique())
        n_cluster_points = datapoints_df.shape[0]
        n_noise_points = noise_df.shape[0]
        min_cluster_size = max(
            int(filtered_df.shape[0]/100) * n_percent,
            min_n_static)

        # change the stats text block
        stats.text = '\n'.join((
                f'Min Cluster Size: {min_cluster_size}',
                f'Epsilon: {distance_in_km*1000}m',
                f'Total Points: {n_cluster_points + n_noise_points}',
                f'Estimated number of clusters: {n_clusters}',
                f'Estimated number of cluster points: {n_cluster_points}',
                f'Estimated number of noise points: {n_noise_points}',
                'Ratio of clustered points to noise: {:0.4f}'.format(
                    float(n_cluster_points)/n_noise_points),
                'Ratio of clustered points to all points: {:0.4f}'.format(
                    float(n_cluster_points)/(n_cluster_points + n_noise_points)),)
                )

    # ==== collect the data ====
    rental_groupby = collect_and_group_data(sys.argv[1])

    # ==== set up widgets =====
    stats = PreText(text='', width=500)

    weekday_selector = RadioButtonGroup(
        labels=list(calendar.day_name),
        active=0,
        name='dow'
    )
    weekday_selector.on_change('active', update)

    hour_selector = RadioButtonGroup(
        labels=[str(x) for x in range(0, 24)],
        active=17,
        name='hour'
    )
    hour_selector.on_change('active', update)

    # ==== initialize ====
    update(None, None, None)

    # set up/draw the map
    map_figure = figure(
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        x_axis_type=map_repr,
        y_axis_type=map_repr,
        title=map_title
    )
    map_figure.add_tile(CARTODBPOSITRON)

    # set up chart
    chart_figure = figure(
        x_range=palette_range,
        plot_height=250,
        title=chart_title
        )
    logging.info(f'Chart range: {chart_figure.x_range.start} {chart_figure.x_range.end}')

    # ==== set up layout ====
    layout = column(
            widgetbox(weekday_selector, hour_selector, width=1000),
            row(map_figure, column(chart_figure, stats)),
        )

    # plot points on map
    plot_points(map_figure, noise_source, datapoints_source)
    # set up the datasource for the chart
    associate_chart(chart_figure, cluster_chart_source)

    curdoc().add_root(layout)
    curdoc().title = 'DOW and Hour Clustering Analysis'

    logging.info('initial map drawn')

main()

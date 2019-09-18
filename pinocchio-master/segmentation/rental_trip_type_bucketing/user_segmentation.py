import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==== functions =====
def haversine_np(lon1, lat1, lon2, lat2):
    """
    from: https://stackoverflow.com/a/29546836
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


@click.command()
@click.argument('infile')
@click.argument('outfile')
@click.option('--rt_radius', default=100,
              help='Maximum parking distance to be considered a round trip')
@click.option('--figfilename', default='rental_type_user_segmentation.png',
              help='Filename of image for rental type histogram')
def main(infile, outfile, rt_radius, figfilename):
    # load csv to df
    df = pd.read_csv(infile)
    # get distance between start and end of rentals
    df['start_end_dist'] = haversine_np(
            df['start_location_lng'],
            df['start_location_lat'],
            df['end_location_lng'],
            df['end_location_lat'])

    # collect the data, groupby preserves order of entry
    customer_df = pd.DataFrame(
            data={
                'round_trip_count': df[df['start_end_dist'] < float(rt_radius)/1000].groupby(['customer_id'])['start_end_dist'].count(),
                'one_way_count': df[df['start_end_dist'] >= float(rt_radius)/1000].groupby(['customer_id'])['start_end_dist'].count(),
                'total': df.groupby(['customer_id'])['start_end_dist'].count()
                },
            index=df.customer_id.unique()
            ).fillna(value=0)

    customer_df = customer_df[customer_df['total'] > 5]
    customer_df['percent_one_way'] = customer_df.apply(lambda x: (x['one_way_count']/x['total'] * 100), axis=1).round(2)
    customer_df['percent_round_trip'] = customer_df.apply(lambda x: (x['round_trip_count']/x['total'] * 100), axis=1).round(2)
    customer_df.to_csv(outfile, index_label='customer_id')

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 6))

    ax1.hist(customer_df['percent_one_way'], bins=list(range(0, 100, 1)), range=(0, 100), log=True)
    ax1.set_xlabel('% one way rentals')
    ax1.set_ylabel('Count')
    ax1.set_title('Histogram of Num of One Way Rentals')
    ax1.grid(True)

    ax2.hist(customer_df['percent_round_trip'], bins=list(range(0, 100, 1)), range=(0, 100), log=True)
    ax2.set_xlabel('% round trip rentals')
    ax2.set_ylabel('Count')
    ax2.set_title('Histogram of Num of Round Trip Rentals')
    ax2.grid(True)

    fig.suptitle(f'{rt_radius}m Min Round Trip Parking Distance (Log Scale)\n', size=16)
    plt.savefig(figfilename)

main()

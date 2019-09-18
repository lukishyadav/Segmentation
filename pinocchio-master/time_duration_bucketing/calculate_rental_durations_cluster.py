import csv
import logging
import json
import pandas as pd
import datetime


def grouper(rental_times_in_seconds_list):
    """
    :param rental_times_in_seconds_list: list of booking durations in seconds
    :return: a grouping of time durations depending on the grouped_time_interval chosen

    this function will group time durations which are {grouped_time_interval} between one another
    """
    prev = None
    group = []
    grouped_time_interval = 1.5

    for item in rental_times_in_seconds_list:
        if not prev or item - prev <= grouped_time_interval:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def cluster_time_durations_booking_start_to_end(rentals_csv_file):
    rental_times_in_seconds = []
    results = {}

    # read from rentals data CSV
    with open(rentals_csv_file, 'r') as csvFile:
        reader = csv.reader(csvFile)
        next(reader)

        for row in reader:
            try:
                # cast start/end/booking start from string to timestamps
                # even though we are only using booking_start_time we cast start time in case we wish to use it
                start_time = datetime.datetime.strptime(row[2].split('.')[0], "%Y-%m-%d %H:%M:%S")
                end_time = datetime.datetime.strptime(row[4].split('.')[0], "%Y-%m-%d %H:%M:%S")
                booking_start_time = datetime.datetime.strptime(row[3].split('.')[0], "%Y-%m-%d %H:%M:%S")

                # get time interval between end_time and booking_start_time and add it to list
                rental_times_in_seconds.append((end_time - booking_start_time).total_seconds())

            except ValueError:
                logging.info("booking_start_time is missing for row {}".format(row))

    # sort the list and create clusters using the grouper() function defined above
    rental_times_in_seconds.sort()
    durations_clusters = (dict(enumerate(grouper(rental_times_in_seconds), 1)))

    for k, v in durations_clusters.items():
        # if a grouping is more than {grouping_amount{} number of items - add cluster to dict
        if len(v) > grouping_amount:

            results["{}-{}".format(min(v), max(v))] = {
                "amount_of_rentals": (len(v))}
        else:
            pass
    return results


# this is the number of items that need to be grouped together for us to consider it a cluster
grouping_amount = 300

# return cluster data
time_duration_cluster_results = cluster_time_durations_booking_start_to_end(rentals_csv_file='darwin_rentals_from_feb1st.csv')
print(time_duration_cluster_results)
# convert to json
r = json.dumps(time_duration_cluster_results)
json_results = pd.read_json(r)

# plot json data
json_results.plot(kind='bar', title="Amount of rentals per time durations cluster", figsize=(15, 10))

import csv
import pandas as pd
from pylab import *

dob_and_rental_count_file = 'dob_and_rental_count.csv'
customer_total_revenue_file = 'customers_ids_with_total_revenue.csv'


def get_age_bucket(age):
    age = age / 2
    return int(age)

def group_rentals_revenue_by_age(customer_dict):
    age_rental_count_dict = {}
    for age, rentals, total_cost in customer_dict.values():
        age_bucket = get_age_bucket(age) * 2
        if age_bucket in age_rental_count_dict.keys():
            age_rental_count_dict[age_bucket]['Total Revenue in USD'] += float(total_cost)
            age_rental_count_dict[age_bucket]['total Rental Count'] += rentals
        else:
            age_rental_count_dict[age_bucket] = {'Total Revenue in USD': float(total_cost),
                                                 'total Rental Count': rentals}
    return age_rental_count_dict

def get_revenue_age_bucket(rental_count_file, total_revenue_file):
    customer_dict = {}

    # iterate through CSV file and add an age and total rental count to customer dictionary
    with open(rental_count_file, 'r') as csvFile:
        reader = csv.reader(csvFile)
        next(reader)
        for row in reader:
            # get customer age by current_time() -  date of birth
            age = int((datetime.datetime.now() - datetime.datetime.strptime(row[1], "%Y-%m-%d")).days / 365.0)
            if age >= 18:
                customer_dict[row[0]] = [age, int(row[2])]

    # iterate through the CSV file and add total charged to the customer dictionary
    with open(total_revenue_file, 'r') as csvFile:
        reader = csv.reader(csvFile)
        next(reader)
        for row in reader:
            if row[0] in customer_dict.keys():
                customer_dict[row[0]] += [(row[1])]

    # if customer doesn't have total cost data add 0.0 instead
    for k, v in sorted(customer_dict.items()):
        if len(v) == 2:
            customer_dict[k] += [0.0]

    # group and sum the rental and total revenue by age buckets
    return group_rentals_revenue_by_age(customer_dict)


df = pd.DataFrame(data=get_revenue_age_bucket(dob_and_rental_count_file, customer_total_revenue_file)).T.sort_index()
df.plot.bar(rot=0, figsize=(15, 10), color=['g', 'y'], grid=True, subplots=True, sharex=False)

savefig("total_cost_total_rentals_by_age_group.png", dpi=100)

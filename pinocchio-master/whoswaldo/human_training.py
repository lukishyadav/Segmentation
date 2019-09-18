import csv
import matplotlib as mpl

mpl.use('TkAgg')

import matplotlib.pyplot as plt
from pandas import read_csv

# csv_file = 'positive_human_validation.csv'  # csv should be sorted by time for a single vehicle and have lat, lng, and rental_id, columns

csv_file = input('filename: ')
df = read_csv(csv_file)
rental_ids = df.rental_id.unique()  # a list of unique rental_ids in the file
with open('human_validation_results.csv', 'w+') as openfile:

    csvwriter = csv.writer(openfile)

    for counter, rental_id in enumerate(rental_ids):  # for each rental_id
        print(rental_id)
        df2 = df[df.rental_id == rental_id]  # slice out just the data for this rental_id
        # plt.subplot(2, 2, counter+1)
        plt.title(rental_id)
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.plot(df2.lng, df2.lat, marker='x')
        plt.show()

        result = input('y/n: ')
        csvwriter.writerow((rental_id,result,))

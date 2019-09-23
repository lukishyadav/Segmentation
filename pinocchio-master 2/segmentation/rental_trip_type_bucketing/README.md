 ## Rental Type Segmentation

 ### What this script does:
 This script intakes rental data and determines the number of round trip rentals and one-way trip rentals.
 
 #### Definitions:
 round trip: a rental where the start and end positions are similar, within a given tolerance of haversine distance.
 one way: a rental where the start and end positions are dissimilar, outside of a given tolerance of haversine distance.
 
 round-trip radius: minimum haversine distance away from rental start position to be considered a round trip. default: 100m
 
 ### Setup:
 In a terminal, perform the following steps:
 
 1. `git clone git@github.com:Ridecell/pinocchio.git`
 1. `cd pinocchio/segmentation/rental_trip_type_bucketing`
 1. `python3 -m venv env`
 1. `source env/bin/activate`
 1. `pip install -r requirements.txt`

 ### Usage: 
 1. If necessary, change the query in "rental\_data\_query".
 2. Log into AWS Athena. Be sure to check your region and the database that you're querying against.
 
 _Please note that as of the time of this writing, there is a data corruption issue with Olympia data and as a result, cannot be queried on Athena_
 
 3. In a terminal: `python user_segmentation.py <infile> <outfile> --figfilename <filename>.png --rt_radius <int>`
 
 #### Options Explanation
 infile: name of csv file retrieved from rental\_data\_query

 outfile: name of file to store csv results

 figfilename: optional output for plot images. Please note that the figures are provided in log scale.

 rt\_radius: optional setting for meters for radius of round trips

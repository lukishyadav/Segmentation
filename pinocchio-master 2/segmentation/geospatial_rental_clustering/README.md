# clustering.py instructions


### Setup

Please note: clustering.py visualization requires python3.7. Get the installation here: https://www.python.org/downloads/release/python-373/

1. Clone the repo: `git clone git@github.com:Ridecell/pinocchio.git`

1. Navigate to this directory: `cd pinocchio/segmentation/geospatial_rental_clustering`

1. In a terminal in this directory: `python3 -m venv ./geospatial_rental_clustering_venv`

1. Enter the environment: `source geospatial_rental_clustering_venv/bin/activate`

1. Install the library for data visualization: `brew install spatialindex`

1. Install the necessary python libraries and packages: `pip install -r requirements.txt`


#### Notes

- The map projection is currently set up to start visualizing over Oakland, CA. To change this, you'll need to change the x\_min, x\_max, y\_min, y\_max values in the map\_constants section of the code.
- If you've made changes to the code or your dataset, you'll have to stop and restart the bokeh server to see those changes reflected.
- If you encounter an error message during the installation process about a package required by Bokeh, you might need zeromq. In your terminal, install this using `brew install zeromq`

## clustering.py
This script works with data from data pipeline. Currently, this requires Athena access.

### Start Visualizing

1. Download some data from datalake via Athena. The sql can be found in the file "datalake\_query". Copy this data with headers intact as a csv and save the data in the same directory as clustering.py. If the headers are not available, use the following header: "rental\_id","customer\_id","start\_datetime","end\_datetime","start\_location\_lat","start\_location\_lng","end\_location\_lat","end\_location\_lng"

1. Start the server: `bokeh serve clustering.py --args <filename>` This might take a while to load. (based on the size of your dataset/size of your datafile)


## clustering\_with\_animation\_db\_source.py
This script works with data from a customer database read-replica.

### Start Visualizing

1. Download some data from the customer database read-replica. The sql can be found in the file "clustering\_analysis\_sql". Copy this data with headers intact as a csv and save the data in the same directory as clustering.py. If the headers are not available, use the following header: "rental\_id","customer\_id","start\_datetime","end\_datetime","start\_location\_lat","start\_location\_lng","end\_location\_lat","end\_location\_lng"

1. Start the server: `bokeh serve clustering\_with\_animation\_db\_source.py --args <filename> <min_n> <min_percent> <epsilon>` This might take a while to load. (based on the size of your dataset/size of your datafile)

    1. filename: name of file you want to use for data.
    1. min\_n: minimum size allowed to be considered a cluster
    1. min\_percent: minimum percentage of all available data points to be considered a cluster
    1. epsilon: max distance between points for cluster determination

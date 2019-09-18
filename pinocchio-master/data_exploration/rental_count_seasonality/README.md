## Seasonality Box Plots

### Purpose
Exploratory analysis of rental count trends per month across specified time slices of a day

### Procedure

#### Setup
1. Get this repo: `git clone git@github.com:Ridecell/pinocchio.git`
1. Change directory to this project's directory: `cd pinocchio/data_exploration/rental_count_seasonality`
1. Create a local environment: `python3 -m venv seasonality_env`
1. Start local environment: `source seasonality_env/bin/activate`
1. Install necessary packages: `pip install -r requirements.txt`

#### Generate Chart
1. Get the data from a customer DB using the query in the file "get\_rental\_registrations\_data". Look for lines labeled: "edit the following/above" as indicators for places to change details pertaining to the customer.
1. Load up jupyter notebook by typing in `jupyter-notebook` in your terminal. This will load up a browser.
1. Using the browser, navigate to the folder "data\_exploration/rental\_count\_seasonality" and click to open the file "rental\_vs\_reg\_seasonality.ipynb".
1. Hit the "Run" button located at the menu bar at the top for each window of code.
1. This should generate a chart in the browser and create a file (unless renamed) called "boxplot.png".

#### Data Analysis: How to read the chart
- error bars = quartiles of data
- box is 2nd and 3rd quartile of data
- green bar across is mean
- red bar across is median
- notch is 85% confidence interval range
- circles outside of error bars are outliers

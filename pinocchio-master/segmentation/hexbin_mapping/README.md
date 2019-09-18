# hexbin visualization instructions


### Setup

Please note: clustering.py visualization requires python3.7. Get the installation here: https://www.python.org/downloads/release/python-373/

1. Clone the repo: `git clone git@github.com:Ridecell/pinocchio.git`

1. Navigate to this directory: `cd pinocchio/segmentation/hexbin_mapping`

1. In a terminal in this directory: `python3 -m venv ./hexbin_mapping_venv`

1. Enter the environment: `source hexbin_mapping_venv/bin/activate`

1. Install the library for data visualization: `brew install spatialindex`

1. Install the necessary python libraries and packages: `pip install -r requirements.txt`

1. Install the jupyter notebook: `pip install jupyter`

1. Start the notebook: `jupyter-notebook`


#### Notes

- The map projection is currently set up to start visualizing over Oakland, CA.
- If you encounter an error message during the installation process about a package required by Bokeh, you might need zeromq. In your terminal, install this using `brew install zeromq`

## Getting Data
This analysis and visualization requires data from data pipeline. Currently, this requires Athena access.

### Data Preprocessing - "hexbin\_data\_preprocessing.ipynb"

1. Download some data from datalake via Athena. The sql can be found in the file "hexbin\_data\_preprocessing.ipynb".
1. Save the data to this directory and change the name references in the cell below "Declare Files Below".
1. Prepare the data by hitting "Cell" at the top menu and selecting "Run All".

### Data Visualization - "hexbin\_visualization.ipynb"

1. Change the reference files in the "Input Files" section.
1. Some defaults are set in the "Analysis Variables" section.
1. Start the visualization by hitting "Cell" at the top menu and selecting "Run All".

#### Write-Up Explaining Project
https://ridecell.quip.com/bGraAaJIiauw/Supply-Demand-Analysis-and-Visualization-Write-Up

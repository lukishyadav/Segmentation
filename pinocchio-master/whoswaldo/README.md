# whoswaldo
Heuristic method to determine nonstandard rentals

More information: https://ridecell.quip.com/vDnbAfuF0xEb/Whos-Waldo

## dependent variables and how they were derived

1. **Duration:**
Difference of time between the last recorded position and the first recorded position
2. **Standard Deviation of Distance between Positions:**
Given a static lat-lng of 0.000000, 0.000000, determine the distance for each point of positional data. 
Calculate the standard deviation of the distances collected.
3. **Distance Traveled:**
Difference between the end odometer reading of the rental and the start odometer reading of the rental.
4. **Farthest Positions Distance:**
Determine the greatest haversine distance between all points of the rental.
5. **Start-End Distance:**
Haversine distance of the position at the beginning of the rental and the end of the rental.
6. **Mean Non-Zero Weight:**
Segment the bounding box of all positions in the rental into bins.
Increment the score of each bin that contains a recorded rental position.
Calculate mean of all bins that have a weight greater than zero.
7. **Standard Deviation of Mean Non-Zero Weight:**
Segment the bounding box of all positions in the rental into bins. 
Increment the score of each bin that contains a recorded rental position.
Calculate standard deviation of all bins that have a weight greater than zero.

## independent variables (definitions of classifications)
1. **standard rental:**
rental mimicking the pattern A-B or A-B-A where A and B are distinct origins or destinations. Ex: commuters, leaving from home for groceries then returning back home
2. **nonstandard rental:**
rentals where the user does not follow the pattern of a nonstandard rental. Ex: multiple destinations, many errands, ridesharing, food couriers

# setup
1. create a virtual environment: `virutalenv env`
2. start the environment: `source env/bin/activate`
3. install the requirements: `pip install -r requirements.txt`

# how to use

## determining rental behavior

### collecting rental data
1. get a list of rental ids as it pertains to a customer (or not)
2. use the following query in splunk. 
   - splunk url: https://splunk.ridecell.com:8000/en-US/app/search/search
   - rental ids in the format "RC-(rental-id) OR RC-(rental-id) ..." 
   ```
   index=darwin-prod event_type=OTAKEYS_VEHICLE_SYNTHESIS funcname=synthesis module=push_views request_method=POST 
   "isEngineRunning\":true" "extId\":\"RC" NOT SDK (**string of rental ids go here**) | spath input=body | 
   rename synthesis.gpsInformation.latitude as lat | where isnum(lat) | 
   rename synthesis.gpsInformation.longitude as lng | where isnum(lng) | 
   rename synthesis.mileageTotal as mileageTotal | where mileageTotal > 0 | rename key.extId as rental_id | 
   table _time, lat, lng, rental_id, mileageTotal |
   sort 0 rental_id, _time 
   ```
3. download the data as a .csv with the filename "customer\_(_id goes here_)\_raw\_data.csv"
4. change the "\_time" heading of the file to be "time".
5. move the file to the base directory of the file

### analyzing the data
1. enter the environment: `source env/bin/activate`
2. enter the ipython shell: `ipython`
3. import the classifier: `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis`
4. import the predicter function: `from main import predictDataset`
5. predict stuff based on your dataset: `predictDataset('training_data.csv', 'customer_30480_raw_data.csv', LinearDiscriminantAnalysis)`

## visualizing random rentals (3 standard, 3 nonstandard)
### note: must have analyzed the data first
1. enter the environment: `source env/bin/activate`
2. enter the ipython shell: `ipython`
3. import the visualization function: `from main import generateRandomResultsViz`
4. run the function: `generateRandomResultsViz(customer_id)`
5. this will place .png files in the code directory, labeled with the rental id

## Possible todos:
- make rental data retrieval easier by calling the splunk api
- make visualization more flexible
- create folders to store rental data csv files

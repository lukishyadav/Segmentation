 ## Calculate Customer RFM Score

 ### What this script does:
 This script intakes customer attributes and calculates a score based on the customer's performance in the service

 #### How is ranking calculated:
 Instead of forcing arbitrary thresholds to determine value, we will be using the data's quantile edges:
* Lower than the 25% average
* Higher than the 25% average lower than 50% average
* Higher than the 50% average lower than 75% average
* Higher than the 75% average

depending on the attribute, having higher or lower values will have positive or negative effects

 ### Setup:
 In a terminal, run the following commands:

 1. `git clone git@github.com:Ridecell/pinocchio.git`
 2. `cd pinocchio/segmentation/calculate_customer_rfm_score`
 3.  `pip install -r requirements.txt`
 4.  `jupyter-notebook`

The jupyter notebook will open in a new browser tab

run the queries against the production database in the user_attributes.query and save the CSV's as "user_att.csv" and "freq_btwn_rentals.csv"

 ### Usage:
 in the jupyter notebook run the script from the top bar.
 The script will output a visualization of the table and download a CSV called 'customer_score.csv' to your local machine



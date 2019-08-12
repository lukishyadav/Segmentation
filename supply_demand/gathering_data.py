#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:14:34 2019

@author: lukishyadav
"""


lat_centerish: 37.8598115
lng_centerish: -122.28253



import requests
import json
payload = {'key': '8a2ea49d9a454d5bb14115114193107', 'q': '37.864197,-122.26588'}
r = requests.get('http://api.apixu.com/v1/current.json', params=payload)
D=json.loads(r.text)


m=list(range(25,32))


import requests
import json

y=0
for x in m:
    

    payload = {'key': '8a2ea49d9a454d5bb14115114193107', 'q': '37.864197,-122.26588','dt':'2019-07-'+str(x)}
    r = requests.get('http://api.apixu.com/v1/history.json', params=payload)
    D=json.loads(r.text)
    
    #D['location']
    DD=D['forecast']
    L=DD['forecastday']
    LL=L[0]
    L[0].keys()
    #http://api.apixu.com/v1/current.json?key=8a2ea49d9a454d5bb14115114193107&q=37.864197,-122.26588
    d=LL['hour']
    
    #d[0]
    import pandas as pd
    
    df=pd.DataFrame.from_dict(d)
    if y==0:
        bigdata=df
        y=1
    else:
        
        bigdata=bigdata.append(df,ignore_index=True)



bigdata.to_csv('/Users/lukishyadav/Desktop/segmentation/supply_demand/last7days.csv',index=False)
#len(LL)




# =============================================================================
# bigdata = df.append(DF, ignore_index=True)
# 
# 
# d1=df.head(1)
# d2=df.head(1)
# 
# 
# bigdata = d1.append(d2, ignore_index=True)
# =============================================================================

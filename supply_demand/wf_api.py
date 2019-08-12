#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:50:18 2019

@author: lukishyadav
"""

import requests
import json
payload = {'key': '8a2ea49d9a454d5bb14115114193107', 'q': '37.864197,-122.26588'}
r = requests.get('http://api.apixu.com/v1/current.json', params=payload)
D=json.loads(r.text)




import requests
import json
payload = {'key': '8a2ea49d9a454d5bb14115114193107', 'q': '37.864197,-122.26588','dt':'2019-07-25'}
r = requests.get('http://api.apixu.com/v1/history.json', params=payload)
D=json.loads(r.text)

D['location']


DD=D['forecast']


L=DD['forecastday']

LL=L[0]

L[0].keys()
#http://api.apixu.com/v1/current.json?key=8a2ea49d9a454d5bb14115114193107&q=37.864197,-122.26588

d=LL['hour']

d[0]

import pandas as pd

df=pd.DataFrame.from_dict(d)

df.to_csv('/Users/lukishyadav/Desktop/segmentation/supply_demand/2019-07-25_lat_lng_37.864197_-122.26588.csv',index=False)
len(LL)
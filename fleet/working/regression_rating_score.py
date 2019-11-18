#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:28:39 2019

@author: lukishyadav
"""

DF['rating_score'].head(10)



DF['rating_score'].shift().head(10)


DF540=DF[DF['vehicle_id']==540]

"""


Segregating by vehicle ids

"""

DF540.sort_values(by=['rental_started_at'],inplace=True)


DF540['i_rating_score']=DF540['rating_score'].shift()


DF540=DF540[['i_rating_score','rating_score']]


DF.drop_duplicates(['rental_id'],keep= 'last',inplace=True)


DF540.dropna(inplace=True)

from sklearn.linear_model import LinearRegression

X=DF540[['i_rating_score']]
y=DF540[['rating_score']]

reg = LinearRegression().fit(X, y)


reg.get_params()

reg.coef_




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:15:46 2019

@author: lukishyadav
"""

import pandas as pd
import my_module
import datetime
import seaborn as sns




df=pd.read_csv('daysfromlastrental_modified_2019-6-25_2145.csv')

"""
Converstion of time delta into no of days (as float)
"""
df['Days from last rental ']=pd.to_timedelta(df['Days from last rental '])

def convert(x):
    return x.total_seconds()/(3600*24)

df['d_f_l_r']=df['Days from last rental '].apply(convert)



# =============================================================================
# DF=pd.read_csv('avg_number_of_days_between_rental_may_2019.csv')
# DF['Average time between rentals']=pd.to_timedelta(DF['Average time between rentals'])
# def convert(x):
#     return x.total_seconds()/(3600*24)
# DF['time']=DF['Average time between rentals'].apply(convert) 
# =============================================================================

DF=pd.read_csv('DarwinRentalCount_modified_2019-7-3_1409.csv')


%matplotlib auto
my_module.perfect_hist(DF['rental_count'])
my_module.fit_distribution(DF['rental_count'],dist=['recipinvgauss','norm','expon'])

DFF=pd.read_csv('DarwinAverageDaysBetweenRental_modified_2019-7-3_1425.csv')
"""
Filtering string not containing - sign
"""
DFF=DFF[~DFF['Average time between rentals'].str.contains("-")]

DFF['Average time between rentals']=pd.to_timedelta(DFF['Average time between rentals'])
def convert(x):
    return x.total_seconds()/(3600*24)
DFF['a_t_b_r']=DFF['Average time between rentals'].apply(convert)

DFF.columns=['customer_id','Average time between rentals','a_t_b_r']

#DFF2=pd.read_csv('Count_greater_than_1_average_days.csv')



Data=pd.merge(DFF[['customer_id','a_t_b_r']],df[['customer_id','d_f_l_r']], on='customer_id', how='inner')

Data=pd.merge(Data[['customer_id','a_t_b_r','d_f_l_r']],DF[['customer_id','rental_count']], on='customer_id', how='inner')

# ['customer_id', 'a_t_b_r', 'd_f_l_r', 'rental_count']


sns.boxplot(x=Data['a_t_b_r'])

m,n,a_t_b_r=my_module.remove_outliers(Data,'a_t_b_r')
m,n,d_f_l_r=my_module.remove_outliers(Data,'d_f_l_r')
m,n,rental_count=my_module.remove_outliers(Data,'rental_count')

DATA=pd.merge(a_t_b_r[['customer_id','a_t_b_r']],d_f_l_r[['customer_id','d_f_l_r']], on='customer_id', how='inner')
DATA=pd.merge(DATA[['customer_id','a_t_b_r','d_f_l_r']],rental_count[['customer_id','rental_count']], on='customer_id', how='inner')

%matplotlib auto
import matplotlib.pyplot as plt
plt.scatter(DATA['a_t_b_r'],DATA['d_f_l_r'])

plt.scatter(DATA['rental_count'],DATA['d_f_l_r'])


plt.scatter(DATA['a_t_b_r'],DATA['d_f_l_r'],DATA['rental_count'])



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =DATA['a_t_b_r']
y =DATA['d_f_l_r']
z =DATA['rental_count']



ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('a_t_b_r')
ax.set_ylabel('d_f_l_r')
ax.set_zlabel('rental_count')

plt.show()
#sns.boxplot(x=a_t_b_r['a_t_b_r'])



"""

Plotting 3d clusters


"""

%matplotlib auto
X=DATA[['a_t_b_r', 'd_f_l_r', 'rental_count']].values
from sklearn.preprocessing import MinMaxScaler
M=MinMaxScaler(feature_range=(0, 1), copy=True)
M.fit(X)
xX=M.transform(X)
 

#xX=DATA[['a_t_b_r', 'd_f_l_r', 'rental_count']].values

from sklearn.cluster import KMeans
algo=KMeans(n_clusters=4, random_state=0)
h=my_module.T_Cluster(xX,algo,x_l='average days between rentals',y_l='days from last rental',z_l='rental_count')




from sklearn.cluster import MeanShift
algo=MeanShift()
algo=MeanShift(bandwidth=0.4839576983534469)
h=my_module.T_Cluster(xX,algo,x_l='average days between rentals',y_l='days from last rental',z_l='rental_count')


from sklearn.cluster import estimate_bandwidth
bw=estimate_bandwidth(xX)



"""

Good DBscan value  (eps=0.05 and min_samples=150)?  (eps=0.05, min_samples=140)

"""

from sklearn.cluster import DBSCAN
algo=DBSCAN(eps=0.05, min_samples=20)
h=my_module.T_Cluster(xX,algo,x_l='average days between rentals',y_l='days from last rental',z_l='rental_count')

algo.KNNdist_plot(xX,50)


"""

Agglomerative Clustering with 5 clusters working decent. (results look biased towards churn)

"""

from sklearn.cluster import AgglomerativeClustering
algo=AgglomerativeClustering(n_clusters=5)
h=my_module.T_Cluster(xX,algo,x_l='average days between rentals',y_l='days from last rental',z_l='rental_count')

"""

Determining CHurn Rule


"""

List=list(h)

LIST=[index for index, value in enumerate(List) if value == 2]

Check=DATA.ix[LIST]    


Check.to_csv('Cluster2.csv',index=False)














%matplotlib auto
X=DATA[['a_t_b_r', 'd_f_l_r']].values
from sklearn.preprocessing import MinMaxScaler
M=MinMaxScaler(feature_range=(0, 1), copy=True)
M.fit(X)
xX=M.transform(X)

#xX=DATA[['a_t_b_r', 'd_f_l_r']].values


from sklearn.cluster import KMeans
algo=KMeans(n_clusters=4, random_state=0)
h=my_module.Cluster(xX,algo,x_l='average days between rentals',y_l='days from last rental')




from sklearn.cluster import AgglomerativeClustering
algo=AgglomerativeClustering(n_clusters=5)
h=my_module.Cluster(xX,algo,x_l='average days between rentals',y_l='days from last rental')




"""
Ideal DBSCAN

"""


from sklearn.neighbors import NearestNeighbors
ns = 20
nbrs = NearestNeighbors(n_neighbors=ns).fit(xX)
distances, indices = nbrs.kneighbors(xX)
distanceDec = sorted(distances[:,ns-1], reverse=True)
plt.plot(list(range(1,len(xX)+1)), distanceDec)
plt.plot(indices[:,0], distanceDec)


import time
start=time.time()
from sklearn.cluster import OPTICS
algo=OPTICS(	min_cluster_size=200)
h=my_module.Cluster(xX,algo,x_l='average days between rentals',y_l='days from last rental')
print(float(time.time()-start)/60.0)





%matplotlib auto
X=DATA[['rental_count', 'd_f_l_r']].values
from sklearn.preprocessing import MinMaxScaler
M=MinMaxScaler(feature_range=(0, 1), copy=True)
M.fit(X)
xX=M.transform(X)

#xX=DATA[['a_t_b_r', 'd_f_l_r']].values


from sklearn.cluster import KMeans
algo=KMeans(n_clusters=4, random_state=0)
h=my_module.Cluster(xX,algo,x_l='Rental_Count',y_l='days from last rental')




from sklearn.cluster import AgglomerativeClustering
algo=AgglomerativeClustering(n_clusters=5)
h=my_module.Cluster(xX,algo,x_l='Rental_Count',y_l='days from last rental')


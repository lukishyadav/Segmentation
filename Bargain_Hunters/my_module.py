#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:09:34 2019

@author: lukishyadav
"""

import pandas as pd
import numpy as np
#import matplotlib
from sklearn.metrics import recall_score,accuracy_score,precision_score
from sklearn.model_selection import KFold
from sklearn import model_selection


# =============================================================================
# def replace_inf(X):
#     DD=DF5['factor2'].copy()
#     DD=DD.replace([np.inf, -np.inf], np.nan).copy()
#     DD.dropna(inplace=True)
#     return DD
# =============================================================================

global outlier
def outlier(DF_F2,x):
    DD=DF_F2[x].copy()
    DD=DD.to_frame()
    Q1 = DD.quantile(0.25)
    Q3 = DD.quantile(0.75)
    IQR = Q3 - Q1 
    UL=(Q3 + 1.5 * IQR)
    LL=(Q1 - 1.5 * IQR)
    DF_F2 = DF_F2[~((DD < (Q1 - 1.5 * IQR)) |(DD > (Q3 + 1.5 * IQR))).any(axis=1)]
    print('IQR:',IQR,'Q1:',Q1,'Q3:',Q3,'UL',(Q3 + 1.5 * IQR))
    return IQR,Q1,Q3,LL,UL,DF_F2


def remove_outliers(X,x):
    OLL=[]
    OUL=[]
    i=0 
    while i<100:
        if i==0:
            a,b,c,d,e,f=outlier(X,x)
            OLL.append(d)
            OUL.append(e) 
            i=i+1
        else:
            a,b,c,d,e,f=outlier(f,x)
            OLL.append(d)
            OUL.append(e)
            if (float(OLL[i-1])!=float(OLL[i])) and (float(OUL[i-1])!=float(OUL[i])):                    
                break
            else:
                i=i+1
    return OLL[-1],OUL[-1],f

def performance_S(model,X,Y,seed):

    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model,X,Y, cv=kfold, scoring='recall')
    print('recall: ',cv_results.mean())
    
    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model,X,Y, cv=kfold, scoring='precision')
    print('precision: ',cv_results.mean())
    
    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model,X,Y, cv=kfold, scoring='accuracy')
    print('accuracy: ',cv_results.mean())

def perfect_hist(X):
    Bins=np.histogram_bin_edges(X,bins='fd')
    
    import matplotlib.pyplot as plt
    plt.clf()
    plt.cla()
    
    plt.hist(X, normed=False, bins=Bins)
    
#['recipinvgauss','norm','expon']
def fit_distribution(X,dist=None):
    from fitter import Fitter
    f = Fitter(X,distributions=dist)
    f.fit()
    f.summary()

def Cluster(xX,algo,x_l='x',y_l='y'):

    km = algo.fit(xX)
    
    Clusters=km.labels_
    
    km.cluster_centers_
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('gnuplot')
    number=len(set(Clusters))
    global color_list
    colors_list = [cmap(i) for i in np.linspace(0, 0.98, number)]
    
    def colors(x):
        if x ==-1:
            return (0.97, 0.970, 0.97, 1.0)    
        else:
            return colors_list[x]
           
    vfunc = np.vectorize(colors)
    CC=vfunc(Clusters)
    
    CCC=list(zip(CC[0],CC[1],CC[2],CC[3]))
    
    CCCC=np.array(CCC)   
    #vfunc = np.vectorize(colors)
    #CC=vfunc(Clusters)
    #%matplotlib auto
    
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    print(set(Clusters))
    
    import collections
    LLL=dict(collections.Counter(Clusters))
    
    plt.figure()
    plt.xlabel(x_l)
    plt.ylabel(y_l) 
    plt.scatter(xX[:,0], xX[:,1],c=CCC)
    C=[]
    for n in list(set(Clusters)):
     C.append(mlines.Line2D([], [], color=colors_list[n], marker='v', linestyle='None',
                              markersize=10, label='Cluster'+str(n)+' : '+str(LLL[n])))
    "_"
    for i in range(len(list(set(Clusters)))):
        plt.legend(handles=C)
            
        
    return Clusters    
        




    
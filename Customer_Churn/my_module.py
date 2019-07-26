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
from sklearn.linear_model import LogisticRegression

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

def Remove_Outliers(X,x,N):
    OLL=[]
    OUL=[]
    i=0 
    while i<N+1:
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
    import warnings
    warnings.filterwarnings("ignore")

    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model,X,Y, cv=kfold, scoring='recall')
    print('recall: ',cv_results.mean())
    
    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model,X,Y, cv=kfold, scoring='precision')
    print('precision: ',cv_results.mean())
    
    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model,X,Y, cv=kfold, scoring='accuracy')
    print('accuracy: ',cv_results.mean())

def performance(model,X,Y,seed):

    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model,X,Y, cv=kfold, scoring='recall')
    print('recall: ',cv_results.mean())
    
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model,X,Y, cv=kfold, scoring='precision')
    print('precision: ',cv_results.mean())
    
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model,X,Y, cv=kfold, scoring='accuracy')
    print('accuracy: ',cv_results.mean()) 

"""

Overriding Logistic regression to make it adjust the threshold for classification 
adding th term to init so that rest of the methods remain same and we have no isse
in using this class with kfold function directly

"""

class LR(LogisticRegression):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='warn', max_iter=100,
                 multi_class='warn', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None,th=0.5):

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.th=th
        
    def predict(self,x):
        Yp=self.predict_proba(x)
        YP=Yp[:,1]
        def decision(x):
            if x > self.th:
                return 1
            else:
                return 0           
        vdecision = np.vectorize(decision)    
        predicted=vdecision(YP) 
        return predicted   



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
    
    #km.cluster_centers_
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
        

def T_Cluster(xX,algo,x_l='x',y_l='y',z_l='z'):
    
#def Cluster(xX,algo,x_l='x',y_l='y'):    

    km = algo.fit(xX)
    
    Clusters=km.labels_
    
    #km.cluster_centers_
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
    
    #plt.figure()
    #plt.xlabel(x_l)
    #plt.ylabel(y_l) 
    #plt.scatter(xX[:,0], xX[:,1],c=CCC)
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x =xX[:,0]
    y =xX[:,1]
    z =xX[:,2]
   
    #ax.scatter(x, y, z, c='r', marker='o')
    ax.scatter(x, y, z, c=CCC)
    
    ax.set_xlabel(x_l)
    ax.set_ylabel(y_l)
    ax.set_zlabel(z_l)
    
    C=[]
    for n in list(set(Clusters)):
     C.append(mlines.Line2D([], [], color=colors_list[n], marker='v', linestyle='None',
                              markersize=10, label='Cluster'+str(n)+' : '+str(LLL[n])))
    "_"
    for i in range(len(list(set(Clusters)))):
        ax.legend(handles=C)
            
        
    return Clusters    



    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 18:51:36 2019

@author: lukishyadav
"""

"""

Method to find out appropriate bin

"""

Bins=np.histogram_bin_edges(C_AMOUNT['Total Credits Used'],bins='fd')

import matplotlib.pyplot as plt
plt.clf()
plt.cla()

plt.hist(C_AMOUNT['Total Credits Used'], normed=False, bins=Bins)
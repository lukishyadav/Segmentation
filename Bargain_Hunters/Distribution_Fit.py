#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:35:45 2019

@author: lukishyadav
"""

"""

Which distribution best fits the data


"""
#, distributions=['gamma', 'rayleigh', 'uniform']

distributions=['recipinvgauss','norm','expon']

from fitter import Fitter
f = Fitter(DF5['days_from_last_rental'],distributions=['recipinvgauss','norm','expon'])
f.fit()
f.summary()
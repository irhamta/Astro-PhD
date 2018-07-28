#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:48:39 2018

@author: irhamta
"""

import numpy as np
import pylab as pl
import pandas as pd
from corner import corner

path = 'COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00/base/plikHM_TTTEEE_lowl_lowE_lensing/'

# MCMC chain samples
samples = np.loadtxt(path + 'base_plikHM_TTTEEE_lowl_lowE_lensing_1.txt')

# load the column names for the samples
column_names = np.loadtxt(path + 'base_plikHM_TTTEEE_lowl_lowE_lensing.paramnames',
                          dtype=np.str, usecols=[0])

# make a data frame with column names and samples
# first two columns are not important
samples1 = pd.DataFrame(samples[:,2:], columns=column_names)

# define which parameters to use
use_params = ['omegam*', 'omegabh2']


#pl.figure()
sigma1 = 1. - np.exp(-(1./1.)**2/2.)
sigma2 = 1. - np.exp(-(2./1.)**2/2.)
_=corner(samples1[use_params], range=[(0.1, 0.5), (0.02, 0.025)], bins=20, levels=(sigma1, sigma2), color='r')


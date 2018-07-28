#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 19:11:36 2018

@author: irhamta
"""

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab

from scipy import ndimage



# =============================================================================
# Plot Supernova Result
# =============================================================================

supernova_data = pd.read_csv('result/output_supernova.csv')


H__, xbins__, ybins__ = pylab.histogram2d(supernova_data['omega_m'], 
                                          supernova_data['omega_l'], 
                                          bins=100)


# =============================================================================
# Delete this part if not necessary
# =============================================================================
#xx = np.random.normal(0.31, 0.000972*5, 45000)
#yy = np.random.normal(0.69, 0.002047*5, 45000)
#
#to_select = (xx + yy < 1.05) & (xx + yy > 0.95)
#xx = xx[to_select]
#yy = yy[to_select]
#
#
#H__, xbins__, ybins__ = pylab.histogram2d(xx, 
#                                          yy, 
#                                          bins=100)

# =============================================================================

H__ = ndimage.gaussian_filter(H__, 2)
sortH__ = np.sort(H__.flatten())
cumH__ = sortH__.cumsum()
lvl68__ = sortH__[cumH__>cumH__.max()*0.32].min()
lvl95__ = sortH__[cumH__>cumH__.max()*0.05].min()
lvl99__ = sortH__[cumH__>cumH__.max()*0.003].min()

c2 = plt.contour(H__.T, [lvl99__, lvl95__, lvl68__], colors='r', 
            extent=(xbins__[0], xbins__[-1], ybins__[0], ybins__[-1]),
            label='Lensing')
c2.collections[0].set_label('Supernova')

''' 
========= From read_supernova_data least-square fitting =========
Least-Square Fitting Final parameter:
Omega_m = 0.31 (0.000972)
Omega_lambda = 0.69 (0.002047)
=================================================================
'''




#delta = 0.025
#xx = np.arange(0, 1, delta)
#yy = np.arange(0, 1, delta)

#_X, _Y = np.meshgrid(xx, yy)
#_Z = mlab.bivariate_normal(_X, _Y, 0.31, 0.69, 0.000972, 0.002047)
#
#c2b = plt.contour(_X, _Y, _Z, colors='r')


# =============================================================================
# Plot Lensing Result
# =============================================================================

lensing_data = pd.read_csv('result/output_lensing.csv')

#lensing_data = lensing_data[(lensing_data.omega_m + lensing_data.omega_l < 1.05)\
#                            & (lensing_data.omega_m + lensing_data.omega_l > 0.95)]



H_, xbins_, ybins_ = pylab.histogram2d(lensing_data['omega_m'], 
                                       lensing_data['omega_l'], 
                                       bins=100)

H_ = ndimage.gaussian_filter(H_, 3)
sortH_ = np.sort(H_.flatten())
cumH_ = sortH_.cumsum()
lvl68_ = sortH_[cumH_>cumH_.max()*0.32].min()
lvl95_ = sortH_[cumH_>cumH_.max()*0.05].min()
lvl99_ = sortH_[cumH_>cumH_.max()*0.003].min()


#quantile = corner.quantile([lensing_data['omega_m'].values, lensing_data['omega_l'].values], 
#                           [0.003, 0.05, 0.32])
#                   
#c1 = corner.hist2d(lensing_data['omega_m'].values, lensing_data['omega_l'].values,
#                   plot_datapoints=False,
#                   plot_density=False,
#                   labels=["$\Omega_m$", "$\Omega_\Lambda$"],
#                   contour_kwargs={'levels': quantile},
#                   color='blue')

c1 = plt.contour(H_.T, [lvl99_, lvl95_, lvl68_], colors='b', 
            extent=(xbins_[0], xbins_[-1], ybins_[0], ybins_[-1]),
            label='Lensing')


c1.collections[0].set_label('Lensing')





# =============================================================================
# Plot CMB data from Planck observation
# =============================================================================


# path to data folder
path = './'

# MCMC chain samples
cmb_data = np.loadtxt(path + 'base_plikHM_TTTEEE_lowl_lowE_lensing_1.txt')

# load the column names for the samples
column_names = np.loadtxt(path + 'base_plikHM_TTTEEE_lowl_lowE_lensing.paramnames',
                          dtype=np.str, usecols=[0])

# make a data frame with column names and samples
# first two columns are not important
planck = pd.DataFrame(cmb_data[:,2:], columns=column_names)


# insert parameters to variable
planck_omegaM = planck['omegam*']
planck_omegaL = planck['omegal*']


H, xbins, ybins = pylab.histogram2d(planck_omegaM ,planck_omegaL, bins=100)

H = ndimage.gaussian_filter(H, 2)
sortH = np.sort(H.flatten())
cumH = sortH.cumsum()
lvl68 = sortH[cumH>cumH.max()*0.32].min()
lvl95 = sortH[cumH>cumH.max()*0.05].min()
lvl99 = sortH[cumH>cumH.max()*0.003].min()

c0 = pylab.contour(H.T, [lvl99, lvl95, lvl68],
              colors='k', extent=(xbins[0], xbins[-1], ybins[0], ybins[-1]),
              label='Planck')
c0.collections[0].set_label('Planck-CMB')



plt.xlabel('$\Omega_m$')
plt.ylabel('$\Omega_\Lambda$')
plt.legend()
plt.xlim(0.2, 0.6)
plt.ylim(0.2, 0.8)
plt.show()
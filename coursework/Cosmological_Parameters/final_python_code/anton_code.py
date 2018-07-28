#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 15:04:09 2018

@author: irhamta
"""
import pandas as pd
import pylab
import numpy as np
import sys
from scipy import ndimage
#sys.path.append('/home/antontj/soft/aum/install/lib/python2.7/site-packages')
#import cosmology as cc

from astropy.cosmology import wCDM, LambdaCDM, FlatwCDM

G = 4.2994e-9       ## Mpc/Msun (Km/s)**2
c = 299792.458     ## km/s

###############################################################################################
## model of cosmology
def cosmolo(Om,w):
#    cosmo=cc.cosmology(Om,0.0,w,0.0,0.0463,0.73,2.726,0.821,0.972,np.log10(8.0),1.0)

#    cosmo = FlatwCDM(Om0=Om, w0=w, H0=70)
    cosmo = LambdaCDM(Om0=Om, Ode0=w, H0=70)    
#    cosmo = LambdaCDM(Om0=Om, w0=w, H0=70.)
    return cosmo
###############################################################################################

###############################################################################################
## cosmology
def D_L(zl,model,Om,w):#angular diameter distance for lens
    return model.angular_diameter_distance(zl) #model.Daofz(zl)


def D_S(zs,model,Om,w):#angular diameter distance for source
    return model.angular_diameter_distance(zs)#Daofz(zs)

def D_LS(zl,zs,model,Om,w):#angular diameter distance between lens and source
    return model.angular_diameter_distance_z1z2(zl, zs)#Daofzlh(zl,zs)

fsize=20

zd = 0.222
zs1 = 0.609
zs2 = 2.4
sig1= 287.
sig2= 97.

fig = pylab.figure()
ax = fig.add_subplot(111)
pylab.subplots_adjust(left=0.1, right=0.99, bottom=0.11, top=0.99)

# now reads in the Planck prior
#f = open('base_w_plikHM_TT_lowTEB_post_lensing_1.txt', 'r')
#f = open('base_plikHM_TT_lowTEB_lensing_1.txt', 'r')

path = 'COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00/base/plikHM_TTTEEE_lowl_lowE_lensing/'

# MCMC chain samples
data = np.loadtxt(path + 'base_plikHM_TTTEEE_lowl_lowE_lensing_1.txt')

# load the column names for the samples
column_names = np.loadtxt(path + 'base_plikHM_TTTEEE_lowl_lowE_lensing.paramnames',
                          dtype=np.str, usecols=[0])

# make a data frame with column names and samples
# first two columns are not important
planck = pd.DataFrame(data[:,2:], columns=column_names)


planck_omegaM = planck['omegam*']
#planck_w = planck['omegabh2']
planck_w = planck['omegal*']

ngrid = 101
omegaM_grid = np.linspace(planck_omegaM.min(), 0.99, ngrid)
#w_grid = np.linspace(planck_w.min(), -1./3., ngrid)
w_grid = np.linspace(planck_w.min(), 1., ngrid)

eta_fiducial1= ((sig1**2*D_LS(zd,zs2,cosmolo(0.25,-1),0.25,-1)+sig2**2*D_LS(zs1,zs2,cosmolo(0.25,-1),0.25,-1))/D_S(zs2,cosmolo(0.25,-1),0.25,-1))/(sig1**2*D_LS(zd,zs1,cosmolo(0.25,-1),0.25,-1)/D_S(zs1,cosmolo(0.25,-1),0.25,-1))
eta_fiducial2= ((sig1**2*D_LS(zd,zs2,cosmolo(0.25,-1),0.25,-1)-sig2**2*D_LS(zs1,zs2,cosmolo(0.25,-1),0.25,-1))/D_S(zs2,cosmolo(0.25,-1),0.25,-1))/(sig1**2*D_LS(zd,zs1,cosmolo(0.25,-1),0.25,-1)/D_S(zs1,cosmolo(0.25,-1),0.25,-1))

#eta_fiducial = Dang(zd, zs1)*Dang(zs2)/Dang(zd, zs2)/Dang(zs1)
delta_eta1 = 0.01*eta_fiducial1
delta_eta2 = 0.01*eta_fiducial2

eta_grid1 = np.zeros((ngrid, ngrid))
eta_grid2 = np.zeros((ngrid, ngrid))

for i in range(ngrid):
    for j in range(ngrid):
        Ds1 =D_S(zs1,cosmolo(Om=omegaM_grid[j],w=w_grid[i]),Om=omegaM_grid[j],w=w_grid[i])
        Ds2 =D_S(zs2,cosmolo(Om=omegaM_grid[j],w=w_grid[i]),Om=omegaM_grid[j],w=w_grid[i])
        Dds1 = D_LS(zd,zs1,cosmolo(Om=omegaM_grid[j],w=w_grid[i]),Om=omegaM_grid[j],w=w_grid[i])
        Dds2 =  D_LS(zd,zs2,cosmolo(Om=omegaM_grid[j],w=w_grid[i]),Om=omegaM_grid[j],w=w_grid[i])
        Ds1s2 = D_LS(zs1,zs2,cosmolo(Om=omegaM_grid[j],w=w_grid[i]),Om=omegaM_grid[j],w=w_grid[i])

        eta_grid1[i, j] = ((sig1**2*Dds2+sig2**2*Ds1s2)/Ds2)/(sig1**2*Dds1/Ds1)
        eta_grid2[i, j] = ((sig1**2*Dds2-sig2**2*Ds1s2)/Ds2)/(sig1**2*Dds1/Ds1)
    like_grid = np.exp(-0.5*(eta_grid1 - eta_fiducial1)**2/delta_eta1**2)#*np.exp(-0.5*(eta_grid2 - eta_fiducial2)**2/delta_eta2**2)
    print (like_grid)

sortH = np.sort(like_grid.flatten())
cumH = sortH.cumsum()

lvl68 = sortH[cumH>cumH.max()*0.32].min()
lvl95 = sortH[cumH>cumH.max()*0.05].min()
lvl99 = sortH[cumH>cumH.max()*0.003].min()

#c1 = pylab.contour(fourlens_like_grid, [lvl68, lvl95, lvl99], colors='r', linestyle='--', extent=(omegaM_grid[0], omegaM_grid[-1], w_grid[0], w_grid[-1]))
c1 = pylab.contourf(like_grid, [lvl68, 1e4], colors='b', alpha=0.9, extent=(omegaM_grid[0], omegaM_grid[-1], w_grid[0], w_grid[-1]))
#c1.collections[0].set_label('Jackpot + Eye of Horus')
c1 = pylab.contourf(like_grid, [lvl95, lvl68], colors='b', alpha=0.5, extent=(omegaM_grid[0], omegaM_grid[-1], w_grid[0], w_grid[-1]))
c1 = pylab.contourf(like_grid, [lvl99, lvl95], colors='b', alpha=0.2, extent=(omegaM_grid[0], omegaM_grid[-1], w_grid[0], w_grid[-1]))

H, xbins, ybins = pylab.histogram2d(planck_omegaM ,planck_w, bins=100)

H = ndimage.gaussian_filter(H, 2)
sortH = np.sort(H.flatten())
cumH = sortH.cumsum()
lvl68 = sortH[cumH>cumH.max()*0.32].min()
lvl95 = sortH[cumH>cumH.max()*0.05].min()
lvl99 = sortH[cumH>cumH.max()*0.003].min()

c0 = pylab.contour(H.T, [lvl68, lvl95, lvl99], colors='k', extent=(xbins[0], xbins[-1], ybins[0], ybins[-1]))
c0.collections[0].set_label('Planck only')

weights = 0.*planck_w
nchain = len(weights)

for i in range(nchain):
    Ds1 = D_S(zs1,cosmolo(Om=planck_omegaM[i],w=planck_w[i]),Om=planck_omegaM[i],w=planck_w[i])
    Ds2 = D_S(zs2,cosmolo(Om=planck_omegaM[i],w=planck_w[i]),Om=planck_omegaM[i],w=planck_w[i])
    Dds1 = D_LS(zd,zs1,cosmolo(Om=planck_omegaM[i],w=planck_w[i]),Om=planck_omegaM[i],w=planck_w[i])
    Dds2 = D_LS(zd,zs2,cosmolo(Om=planck_omegaM[i],w=planck_w[i]),Om=planck_omegaM[i],w=planck_w[i])
    Ds1s2= D_LS(zs1,zs2,cosmolo(Om=planck_omegaM[i],w=planck_w[i]),Om=planck_omegaM[i],w=planck_w[i])
    eta1 = ((sig1**2*Dds2+sig2**2*Ds1s2)/Ds2)/(sig1**2*Dds1/Ds1)
    eta2 = ((sig1**2*Dds2-sig2**2*Ds1s2)/Ds2)/(sig1**2*Dds1/Ds1)


    weights[i] = np.exp(-0.5*(eta1 - eta_fiducial1)**2/delta_eta1**2)*np.exp(-0.5*(eta2 - eta_fiducial2)**2/delta_eta2**2)

H, xbins, ybins = pylab.histogram2d(planck_omegaM, planck_w, weights=weights, bins=100)

H = ndimage.gaussian_filter(H, 2)
sortH = np.sort(like_grid.flatten())
cumH = sortH.cumsum()
lvl68 = sortH[cumH>cumH.max()*0.32].min()
lvl95 = sortH[cumH>cumH.max()*0.05].min()
lvl99 = sortH[cumH>cumH.max()*0.003].min()

c2 = pylab.contour(H.T, [lvl68, lvl95, lvl99], colors='r', linewidths=3, extent=(xbins[0], xbins[-1], ybins[0], ybins[-1]))
c2.collections[0].set_label('DLSP + Planck')

pylab.xlabel('$\Omega_M$', fontsize=fsize)
pylab.ylabel('$w$', fontsize=fsize)
pylab.xticks(fontsize=14)
pylab.yticks(fontsize=14)

xlim = pylab.xlim()
ylim = pylab.ylim()

pylab.plot(np.arange(10), np.zeros(10) - 1e10, color='b', linewidth=10, label='DLSP')
#box = ax.get_position()
pylab.xlim(xlim[0], xlim[1])
pylab.ylim(-2., -1./3.)


#box = ax.get_position()

#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45))
pylab.legend(loc='upper right')

pylab.savefig('cosmology_forecast.png')
#pylab.show()
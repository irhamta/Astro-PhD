#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 08:33:45 2018

@author: evaria
"""

# impoort modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.cosmology import FlatLambdaCDM, Planck15
from scipy import integrate, optimize

# get Planck Collaboration et al. (2015) cosmological data for model benchmark
cosmo = Planck15


# =============================================================================
# Data cleaning
# =============================================================================

# read json data
data = pd.read_json('supernova.json')

# define list of strings to be removed
bad_string = '(host|spectroscopic|heliocentric|cmb|photometric|cluster|,)'

# define list of columns to be removed
bad_column = ['z', 'dL (Mpc)', 'mmax', 'Mmax']

# replace z and dL value with valid float numbers
# nan means Not A Number
for i in bad_column:
    data[i] = data[i].str.replace(bad_string, '')
    data[i].loc[data[i] == ''] = np.nan
    data[i] = data[i].astype(float)

# sort data by redshift value
data = data.sort_values(by=['z'])

# redshift cut until z~2
data = data.loc[data['z'] <= 2]

# plot redshift vs distance modulus
#plt.plot(data['z'], data['mmax'] - data['Mmax'], 'b.')
#plt.plot(data['z'], cosmo.distmod(data['z']), 'k--')


# =============================================================================
# Least-square Fitting
# =============================================================================


# define cosmological parameters
c = 3 * 1e5 # km/s
H0 = 67.7 #km / (Mpc s)
Omega_m = 0.307
Omega_r = 0 * 1e-5 # too small
Omega_lambda = 1 - Omega_m

# make luminosity distance function
def lum_dist (z, Omega_m, Omega_lambda, H0):

    Omega_r = 0 * 1e-5 # too small

    # integration part
    # integration is calculated from redshift=0 to redshift=z
    fn = lambda z: (Omega_r*(1+z)**4. + Omega_m*(1+z)**3 + Omega_lambda)**-0.5

    # return array values
    return c*(1+z)/H0 * np.asarray([integrate.quad(fn, 0, _z)[0] for _z in z])

# calculate luminosity distances
dL = lum_dist(data['z'], Omega_m, Omega_lambda, H0)


# remove NaN values
data_good = data[['z', 'dL (Mpc)']].dropna()
data_good = data_good.sample(n=100)

# guess initial parameters
initial_param = np.array([Omega_m, Omega_lambda, H0])

# least-square fitting
opt_param, cov = optimize.curve_fit(lum_dist, 
                                    data_good['z'].values, 
                                    data_good['dL (Mpc)'].values, 
                                    p0=initial_param)
err_param = np.sqrt(np.diag(cov))

# =============================================================================
# Plot the result
# =============================================================================

plt.figure()
plt.plot(data['z'], data['dL (Mpc)'], 'b.', label='Data')
plt.plot(data['z'], cosmo.luminosity_distance(data['z']), 'k--', label='Astropy')
plt.plot(data['z'], dL, 'r--', label='lum_dist function')
plt.plot(data['z'], lum_dist(data['z'], *opt_param), 
         'g-', label='lum_dist function (optimized)')
plt.ylabel('Distance (Mpc)')
plt.xlabel('Redshift')
plt.legend()
plt.close('all')

print ('Least-Square Fitting Final parameter:')
print ('Omega_m = %.2f (%f)' %(opt_param[0], err_param[0]))
print ('Omega_lambda = %.2f (%f)' %(opt_param[1], err_param[1]))


# =============================================================================
# Maximum likelihood fitting
# =============================================================================


# define likelihood function as in Equation 11 in Leaf et al. (2018)
def lnlike(theta, X, y, yerr):

    Omega_m, Omega_lambda, H0 = theta
    model = lum_dist(X, Omega_m, Omega_lambda, H0)

    # chi-square
    chi2 = ((y-model)**2)/yerr**2

    return np.sum(1/(np.sqrt(2*np.pi)*yerr) * (np.exp(-chi2/2)))

#    return np.sum( np.exp(-chi2/2) )

X = data_good['z'].values
y = data_good['dL (Mpc)'].values
yerr = 0.2

from scipy import optimize

# optimize module minimizes functions whereas we would like to maximize the likelihood
# that's why I put the minus(-) sign
nll = lambda *args: -lnlike(*args)
result = optimize.minimize(nll, [Omega_m, Omega_lambda, H0], args=(X, y, yerr))
m_ml, b_ml, h0_ml = result["x"]

print ('======================================')
print ('Maximum Likelihood Result')
print ('Omega_m = %.2f (%.2f)' %(m_ml, 0))
print ('Omega_lambda = %.2f (%.2f)' %(b_ml, 0))
print ('H0 = %.2f (%.2f)' %(h0_ml, 0))
print ('======================================\n')


# =============================================================================
# MCMC fitting
# see http://dfm.io/emcee/current/user/line/ for the detail
# =============================================================================

# define prior
def lnprior(theta):
    Omega_m, Omega_lambda, H0 = theta
    if 0 < Omega_m < 1 and 0 < Omega_lambda < 1:
        return 0
    return -np.inf

# define the full probability
def lnprob(theta, X, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, X, y, yerr)


# 
ndim, nwalkers = 3, 100
#pos = [result["x"] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

pos = [[Omega_m, Omega_lambda, H0] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

import emcee
import sys

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(X, y, yerr), threads=3)

nsteps = 500
width = 30

print ('running MCMC.....')
for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
    n = int((width+1) * float(i) / nsteps)
    sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
sys.stdout.write("\n")

samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

import corner
fig = corner.corner(samples, labels=["$\Omega_m$", "$\Omega_\Lambda$", "$H_0$"],
                      truths=[Omega_m, Omega_lambda, H0])

plt.savefig('result/supernova.png')
plt.show()



m_mcmc, b_mcmc, h0_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                              zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

print ('======================================')
print ('MCMC Result')
print ('Omega_m = ', m_mcmc)
print ('Omega_lambda = ', b_mcmc)
print ('H0 = ', h0_mcmc)
print ('======================================\n')


output_data = pd.DataFrame({'omega_m': samples[:, 0], 
                            'omega_l': samples[:, 1],
                            'h0'     : samples[:, 2]})
output_data.to_csv('result/output_supernova.csv', index=False)
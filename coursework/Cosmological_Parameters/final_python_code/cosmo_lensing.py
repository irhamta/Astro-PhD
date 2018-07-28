#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 11:08:12 2018

@author: irhamta

Based on Leaf & Melia (2018) paper
"""

# import necessary modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# read lensing data which is taken from Leaf et al. (2018)
data = pd.read_csv('lensing_data.csv')

data = data[data.name != 'J0850-0347'] # outlier removal

# define function to calculate angular diameter distance as in Equation 3
# this is used for observational data
def D_obs(theta_E, sigma_0):
    
    # need to convert theta_E from arcsec to radian    
    theta_E = np.deg2rad(theta_E/3600)
    
    # speed of light in km/s
    c = 3*1e5 
    
    return c**2 * theta_E / (4*np.pi * sigma_0**2)

# error of observed angular diameter distance as in Equation 10
# this doesn't include the sigma_x
def D_obs_err(theta_E, err_theta_E, sigma_0, err_sigma_0):
    return D_obs(theta_E, sigma_0) \
            * np.sqrt((err_theta_E/theta_E)**2 + (2*err_sigma_0/sigma_0)**2 + 0.12**2)



# =============================================================================
# This part is to make sure that our written D_obs function is consistent with
# built in function from astropy
# =============================================================================

from astropy.cosmology import LambdaCDM
import astropy.units as u

# cosmological model to calculate cosmological parameters
def cosmolo(Om0,Ode0):
    cosmo = LambdaCDM(Om0=Om0, Ode0=Ode0, H0=67.7, Tcmb0=2.725,
                      Ob0=0.0486, m_nu=u.Quantity([0., 0., 0.06], u.eV))
    return cosmo

# angular diameter distance for lens
def D_L(zl,model):
    return model.angular_diameter_distance(zl)

#angular diameter distance for source
def D_S(zs,model):
    return model.angular_diameter_distance(zs)

#angular diameter distance between lens and source
def D_LS(zl,zs,model):
    return model.angular_diameter_distance_z1z2(zl, zs)


# calculate the angular diameter distance
Dobs = D_obs(data['theta_E'], data['sigma_0'])
Dobs_err = D_obs_err(data['theta_E'], data['theta_E']*0.05, data['sigma_0'], data['err_sigma_0'])
Dteo = D_LS(data['zl'], data['zs'], cosmolo(Om0=0.3, Ode0=0.7))/D_S(data['zs'], cosmolo(Om0=0.3, Ode0=0.7))

# uncomment these plots if you need to do some validation
#plt.plot(Dobs, Dteo, 'b.')
#plt.plot(data['sigma_0'], Dteo, 'r.')
#plt.plot(data['sigma_0'], Dobs, 'b.')



# =============================================================================
# Build some functions to calculate theoretical angular diameter distance
# =============================================================================

# cut outliers based on page 3 in the paper
data = data[Dobs < 1]

# recalculate
Dobs = D_obs(data['theta_E'], data['sigma_0'])
Dobs_err = D_obs_err(data['theta_E'], data['theta_E']*0.05, data['sigma_0'], data['err_sigma_0'])




from scipy import integrate

# define cosmological parameters
c = 3 * 1e5 # km/s
H0 = 67.7 #km / (Mpc s)
Omega_m = 0.307
Omega_r = 0 * 1e-5 # too small
Omega_lambda = 1 - Omega_m
wde = -1

# make angular diameter distance calculator
# equivalent to cosmolo.angular_diameter_distance_z1z2()
# based on Equation 4 in Leat et al. (2018)
def ang_dist (z_1, z_2, Omega_m, Omega_lambda, wde):
    # integration part
    # integration is calculated from redshift=zl to redshift=zs
    fn = lambda z: (Omega_r*(1+z)**4. \
                    + Omega_m*(1+z)**3 \
                    + Omega_lambda*(1+z)**(3*(1+wde)) \
                    )**-0.5

    # return array values
    return c/(H0*(1+z_2)) \
        * np.asarray([integrate.quad(fn, _z[0], _z[1])[0] for _z in list(zip(z_1, z_2))])


# =============================================================================
# Validation for ang_dist() function
# =============================================================================
#DS = ang_dist(data['zl'], data['zs'], 0.3, 0.7)
#plt.plot(DS, D_LS(data['zl'], data['zs'], cosmolo(Om0=0.3, Ode0=0.7)), 'b.')
# =============================================================================


# D theoretical based on Equation 7 in Leaf et al. (2018)
def D_theory(X, Omega_m, Omega_lambda, wde):
    z_1, z_2 = X
    return ang_dist(z_1, z_2, Omega_m, Omega_lambda, wde) \
            / ang_dist(0*z_1, z_2, Omega_m, Omega_lambda, wde)

# =============================================================================
# Validation for ang_dist() function
# =============================================================================
# to validate D_theory() function
#plt.plot(Dteo, D_theory((data['zl'], data['zs']), 0.3, 0.7), 'b.')
# =============================================================================



# =============================================================================
# Maximum likelihood fitting
# =============================================================================


# define likelihood function as in Equation 11 in Leaf et al. (2018)
def lnlike(theta, X, y, yerr):

    Omega_m, Omega_lambda, wde = theta
    model = D_theory(X, Omega_m, Omega_lambda, wde)

    # chi-square
    chi2 = ((y-model)**2)/yerr**2

    return np.sum(1/(np.sqrt(2*np.pi)*yerr) * (np.exp(-chi2/2)))


X = (data['zl'].values, data['zs'].values)
y = Dobs
yerr = Dobs_err

from scipy import optimize

# optimize module minimizes functions whereas we would like to maximize the likelihood
# that's why I put the minus(-) sign
nll = lambda *args: -lnlike(*args)
result = optimize.minimize(nll, [Omega_m, Omega_lambda, wde], args=(X, y, yerr))
m_ml, b_ml, wde_ml = result["x"]

print ('======================================')
print ('Maximum Likelihood Result')
print ('Omega_m = %.2f (%f)' %(m_ml, 0))
print ('Omega_lambda = %.2f (%f)' %(b_ml, 0))
print ('w_de = %.2f (%f)' %(wde_ml, 0))
print ('======================================\n')




# =============================================================================
# MCMC fitting
# see http://dfm.io/emcee/current/user/line/ for the detail
# =============================================================================

# define prior
def lnprior(theta):
    Omega_m, Omega_lambda, wde = theta
    if 0.1 < Omega_m < 0.9 \
    and 0.1 < Omega_lambda < 0.9 \
    and -2 < wde < 1:
        return 0
    return -np.inf

# define the full probability
def lnprob(theta, X, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, X, y, yerr)


# 
ndim, nwalkers = 3, 300
#pos = [result["x"] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

pos = [[Omega_m, Omega_lambda, wde] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

import emcee
import sys


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(X, y, yerr), threads=3)

nsteps = 1000
width = 30

print ('running MCMC.....')
for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
    n = int((width+1) * float(i) / nsteps)
    sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
sys.stdout.write("\n")


samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

import corner
fig = corner.corner(samples, labels=["$\Omega_m$", "$\Omega_\Lambda$", "$w_{\\rm de}$"],
                      truths=[Omega_m, Omega_lambda, wde])

plt.savefig('result/lensing.png')
plt.show()



m_mcmc, b_mcmc, wde_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

print ('======================================')
print ('MCMC Result')
print ('Omega_m = ', m_mcmc)
print ('Omega_lambda = ', b_mcmc)
print ('w_de = ', wde_mcmc)
print ('======================================\n')


output_data = pd.DataFrame({'omega_m': samples[:, 0], 
                            'omega_l': samples[:, 1],
                            'wde'    : samples[:, 2]})
output_data.to_csv('result/output_lensing.csv', index=False)
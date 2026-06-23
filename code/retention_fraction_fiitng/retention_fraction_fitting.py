# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:29:23 2024

@author: Kecsi
"""

"""
Reads in data for retention fraction plot, makes fit plots.
"""


import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


import scienceplots
plt.style.use(['science','no-latex'])


SAMPLE_SIZE = 10000

data = np.loadtxt(f"raw_{SAMPLE_SIZE}_retention_fraction.csv",delimiter=',')

kicks = data[:,0]
retention = data[:,1]

kicks = np.insert(kicks,0,0)
retention = np.insert(retention,0,1)
#Rescale kicks
#kicks /= 750.

def fermi_dirac(x,c,x_fermi,beta):
    
    return (c/(np.exp((x-x_fermi)*beta) + 1))

def third_polynomial(x,a,b,c,d):
    
    return (a*x**3+b*x**2+c*x+d)

#Fermi fit
#params, cov = curve_fit(fermi_dirac,kicks,retention)
#c, x_f, b = params

#Third order polynomial
params, cov = curve_fit(third_polynomial,kicks,retention)
a, b, c, d = params

np.savetxt('10kpolyfit_with_0.csv', params, delimiter=',')
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)

dummy_v = []
for i in range(751):
    dummy_v.append(i)

dummy_v = np.array(dummy_v)
#ax.scatter(kicks,retention,label='Simulated Retention Fraction',marker='x')
#ax.plot(kicks, fermi_dirac(kicks,c, x_f,b),color='red',label='Fermi-Dirac distribution')
ax.errorbar(kicks, retention, yerr=(np.sqrt(retention/SAMPLE_SIZE)),label='Simulated Retention Fraction',marker='.')
ax.plot(dummy_v, third_polynomial(dummy_v, a, b, c, d), color='red', label=r'Polynomial of O(x$^3$)')
ax.set_xlabel(r'Kick velocity (kms$^{-1}$)',fontsize=16)
ax.set_ylabel('Retention Fraction',fontsize=16)
ax.set_title(rf"Simulated Retention Fraction",fontsize=20)
ax.legend(loc=1)
#plt.savefig(rf"{SAMPLE_SIZE}_retention_fraction_3poly.pdf")
plt.show()

#%%

import scipy.integrate as integrate

OVERALL_SCALE = 286

sigma1 = 75
sigma2 = 316
scale1 = 0.42
scale2 = 0.58



def maxwell(x, s):
    
    return np.sqrt(2/np.pi)*((x**2)/s**3)*np.exp(-(x**2/(2*s**2)))


def bi_maxwell(x, s1, s2, f1, f2):
    
    return (f1*maxwell(x,s1) + f2*maxwell(x,s2))


def minus_1(x, s1, s2, f1, f2, a, b, c, d):
    
    return (f1*maxwell(x,s1) + f2*maxwell(x,s2))*(1-(a*x**3+b*x**2+c*x+d))


def minus_2(x, s1, s2, f1, f2, a, b, c, d):
    
    return (f1*maxwell(x,s1) + f2*maxwell(x,s2))*(2-(a*x**3+b*x**2+c*x+d))


def over(x, s1, s2, f1, f2, a, b, c, d):
    
    return (f1*maxwell(x,s1) + f2*maxwell(x,s2))/((a*x**3+b*x**2+c*x+d))


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

dummy_k = []
for i in range(0,500, 10):
    dummy_k.append(i)
    
dummy_k = np.array(dummy_k)

new_kicks = dummy_k
new_ret = third_polynomial(dummy_k, a, b, c, d)





dummy_x = []
for i in range(1001):
    dummy_x.append(i)
    
dummy_x = np.array(dummy_x)

normal = integrate.quad(third_polynomial, 0, 1000, args=(a, b, c, d))




ax.plot(dummy_x, OVERALL_SCALE*bi_maxwell(dummy_x, sigma1, sigma2, scale1, scale2), color='black', linewidth=2, label='Observed velocity distribution $f(v)$')


#ax.plot(dummy_x, 1-third_polynomial(dummy_x, a, b, c, d), color='green', label=r'Polynomial of O(x$^3$)')

#ax.plot(dummy_x, OVERALL_SCALE*minus_1(dummy_x, sigma1, sigma2, scale1, scale2, a, b, c, d), color='green', label=r'Intrinsic distribution 1: $f(v)(1-g(v))$')
#ax.plot(dummy_x, OVERALL_SCALE*minus_2(dummy_x, sigma1, sigma2, scale1, scale2, a, b, c, d), color='red', label = r'Intrinsic distribution 2: $f(v)(2-g(v))$')


ax.plot(dummy_k, OVERALL_SCALE*bi_maxwell(new_kicks, sigma1, sigma2, scale1, scale2)/new_ret, color='red', label = r'Intrinsic distribution guess: $\frac{f(v)}{g(v)}$')
ax.plot(dummy_x, third_polynomial(dummy_x, a, b, c, d), color='green', label='Scaled retention fraction',alpha=0.5)

ax.axvline(450., color='blue', linestyle='--', linewidth=2, label='Reliability limit', alpha=0.5)
ax.fill_between(dummy_x[449:], y1=OVERALL_SCALE*0.004, alpha = 0.25)

ax.set_xlim(xmin=0., xmax=600.)
ax.set_ylim(ymin=0., ymax=OVERALL_SCALE*0.004)
ax.set_title('Velocity distribution of pulsars', fontsize=20)
ax.set_xlabel(r'Velocity (kms$^{-1})$', fontsize=16)
ax.set_ylabel('Probability density x Overall Scale (286)', fontsize=16)
plt.legend(fontsize=14)
plt.savefig('intrinsic_velocity_distribution_pulsar_3.pdf')
plt.show()
#%%
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

#new_kicks = kicks[:11]
#new_ret = retention[:11]

ax.plot(dummy_x, OVERALL_SCALE*bi_maxwell(dummy_x, sigma1, sigma2, scale1, scale2)/(third_polynomial(dummy_x, a, b, c, d)/normal[0]), color='yellow')

plt.show()
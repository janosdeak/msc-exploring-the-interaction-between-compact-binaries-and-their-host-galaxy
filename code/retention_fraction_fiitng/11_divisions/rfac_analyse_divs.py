# -*- coding: utf-8 -*-
"""
Created on Sat May  4 14:42:25 2024

@author: Kecsi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


SAMPLE_SIZE = 10000


def third_polynomial(x,a,b,c,d):
    
    return (a*x**3+b*x**2+c*x+d)

def fermi_dirac(x,c,x_fermi,beta):
    
    return (c/(np.exp((x-x_fermi)*beta) + 1))

drop_points = []


for i in range(1,17):
    data = np.loadtxt(f"raw_{SAMPLE_SIZE}_retention_fraction_R{i}.csv",delimiter=',')
    
    kicks = data[:,0]
    retention = data[:,1]
    
    kicks = np.insert(kicks,0,0)
    retention = np.insert(retention,0,1)
    
    #Fitting for each dataset
    #params, cov = curve_fit(third_polynomial,kicks,retention)
    #a, b, c, d = params
    
    kicks /= 750.
    params, cov = curve_fit(fermi_dirac,kicks,retention)
    c, x_f, b = params
    
    np.savetxt(f"{SAMPLE_SIZE}_polyfit_with_0_R{i}.csv", params, delimiter=',')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    dummy_x =[]
    for j in range(101):
        dummy_x.append(j/100)
        
    k = 0
    while (fermi_dirac(dummy_x[k], c, x_f, b) > 0.5):
        k += 1
        
    ax.scatter(kicks,retention,label='Simulated Retention Fraction',marker='x')
    ax.plot(dummy_x, fermi_dirac(dummy_x,c, x_f,b),color='red',label='Fermi-Dirac distribution')
    ax.axvline(dummy_x[k], color='green', linestyle='--', label='Inflection point x')
    ax.axhline(fermi_dirac(dummy_x[k],c, x_f,b), color='blue', linestyle='--', label='Inflection point y' )
    #ax.plot(kicks, third_polynomial(kicks, a, b, c, d), color='red', label=r'Polynomial of O(x$^3$)')
    ax.set_xlabel(r'Kick velocity (kms$^{-1}$)',fontsize=16)
    ax.set_ylabel('Retention Fraction',fontsize=16)
    if (i==16):
        ax.set_title(f"Retention fraction for systems born outside of 15 kpc", fontsize=20)
    else:
        ax.set_title(rf"Rentention fraction for systems born between {i-1}-{i} kpc",fontsize=20)
    ax.legend(loc=1)
    plt.savefig(rf"{SAMPLE_SIZE}_retention_fraction_3poly_R{i}.pdf")
    

    drop_points.append([i,dummy_x[k]*750.])
    
drop_points = np.delete(drop_points, 15,axis=0)
    
np.savetxt('drop_below50_points.csv',drop_points,delimiter=',')
#%%
drop_points = np.array(drop_points)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)



def negexp(x, a, b, c, d):
    
    return (a*np.exp(c*(x+b)) + d)

params, cov = curve_fit(negexp,drop_points[:,0], drop_points[:,1])
d, e, f, g = params

dummy_r = []
for i in range(161):
    dummy_r.append(i/10)

ax.scatter(drop_points[:,0], drop_points[:,1], label='Data points', marker='x')
ax.plot(dummy_r, negexp(dummy_r, d, e, f, g), color='red', label='Exponential fit')

ax.set_ylim(bottom=0.)
ax.set_xlabel(r'Radius slices (kpc)',fontsize=16)
ax.set_ylabel(r'Kick velocity (kms$^{-1})$',fontsize=16)
ax.set_title('Points when the retention fraction drops below 50%',fontsize=20)
plt.legend()
#plt.savefig(f"retention_fraction_{SAMPLE_SIZE}dropping_below_50.pdf")
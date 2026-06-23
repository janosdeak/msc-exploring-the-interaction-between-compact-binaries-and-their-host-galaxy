# -*- coding: utf-8 -*-

"""
Created on Tue Nov 28 16:39:44 2023

@author: Jánosdeák Márk
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rv_continuous
from scipy.interpolate import UnivariateSpline

from astropy import units as u
from galpy.orbit import Orbit
from galpy.potential import vcirc, MWPotential2014
from galpy.potential import MiyamotoNagaiPotential, PowerSphericalPotentialwCutoff, mass
# Constants:

FILE_NAMES = ['input_data/NSNS_met_0-02_BPASS_processed.csv','input_data/NSNS_met_0-004_BPASS_processed.csv','input_data/NSNS_met_0-008_BPASS_processed.csv']
SET_DATA = 0 # Set which data file do you want to work with
SAMPLE_SIZE = 100 # The number of binary stars you wish to work with.
BIN_NUMBER = 100 # Number of bins when binning the enclosed mass function
GALACTIC_RADIUS = 27. # The radius of the galaxy's baryonic matter(1%)
HUBBLE_TIME = 14. # Hubble Time in Gyr
KICK_VELOCITY_MAGNITUDES = 50. # For the custom kicks
INTEGRATION_STEP_1 = 100
INTEGRATION_STEP_2 = 1000


def file_validation():
    """
    Checks if data directories exists, if not print a message to the user.
    If both of the files are found it returns True otherwise False.

    Returns:
    -------
    boolean
    """

    if not (os.path.exists(FILE_NAMES[0]) or os.path.exists(FILE_NAMES[1]) or os.path.exists(FILE_NAMES[2])):
        print("All data files not found or have an unsufficient names. Please "
              "rename them to: "
              + FILE_NAMES[0] + ' and ' + FILE_NAMES[1] + ' and ' + FILE_NAMES[2] +'\n')
        return False

    if not os.path.exists(FILE_NAMES[0]):
        print("First data file not found or have an unsufficient name. Please "
              "rename it to: " + FILE_NAMES[0] + '\n')
        return False

    if not os.path.exists(FILE_NAMES[1]):
        print("Second data file not found or have an unsufficient name. Please"
              " rename it to: " + FILE_NAMES[1] + '\n')
        return False

    if not os.path.exists(FILE_NAMES[2]):
        print("Third data file not found or have an unsufficient name. Please"
              " rename it to: " + FILE_NAMES[2] + '\n')
        return False

    return True
    
    
def read_input(file_names=FILE_NAMES):
    """
    Read input and save to a data file. Utilize panda package.
    
    Args:
    -------
    file_names : string or array of string
    
    Returns:
    -------
    data:  2D numpy array with axis 0 being a lot of data entries and axis 1 is
        the following: ID, ZAMS_1, ZAMS_2, Age_1, Age_2, Kikc_1, Kick_2, Lifespan.
        Where ZAMS: Zero Age Mean Sequence (i.e. mass of star at birth), Age_N:
        The time Kick number N happens, Lifespan: time when merging happens.
    """
    
    data_frame = pd.read_csv(file_names[SET_DATA])
    data = data_frame.to_numpy()
    
    return data
    

def sample_data(data):
    """
    Selectes a number of entries from the original data file using randomness.
    The selected number equals SAMPLE_SIZE
    
    Args:
    ----------
    data : 2D numpy array
    
    Returns:
    -------
    selected_data: Array of shape (SAMPLE_SIZE, 8)
    """    
    
    # Creates the random indices for the data file, replace=False means an entry wont be repeated
    sample_indicies = np.random.choice(data.shape[0], size=SAMPLE_SIZE, replace=False)
    selected_data = data[sample_indicies]
    
    return selected_data


def galactic_potential():
    """
    Creating a galactic potential without a dark matter halo. i.e for baryonic
    matter only. (Potnetially look for what determines star formation rates and,
    how to distribute binaries.)

    Returns:
    -------
    galpy potential object
    """
    
    # Normalization set for overall 1, used normalization in MWPotential2014 as reference
    mn = MiyamotoNagaiPotential(a=3./8.,b=0.28/8.,normalize=(.6/.65))
    bp= PowerSphericalPotentialwCutoff(alpha=1.8,rc=1.9/8.,normalize=(.05/.65))

    mw14_baryonic = mn + bp

    return mw14_baryonic
    

def binning_potential(potential):
    """
    Binns the enclosed mass function to get a histogram of mass in a range dR.
    Determines the mid values for the dR ranges.
    
    Args:
    -------
    potential: galpy potential object
    
    Returns:
    -------
    mass_z0: the binned mass for a dR range, numpy array
    r_mid: the midpoint of the dR range, numpy array
    """
    
    r_val = np.linspace(.036,GALACTIC_RADIUS/8.,BIN_NUMBER)
    
    r_mid = r_val + (r_val[1] - r_val[0])/2.
    
    mass_encl = np.empty(BIN_NUMBER)
    
    for i in range(BIN_NUMBER):
        mass_encl[i] = mass(potential, R=r_val[i], z=10.)
        
    mass_z0 = np.empty(BIN_NUMBER)
    mass_z0[0] = mass_encl[0]
    
    for i in range(BIN_NUMBER-1):
        mass_z0[i+1] = (mass_encl[i+1] -  mass_encl[i])
    
    return mass_z0, r_mid


def fitting_to_binned_mass(x_data, y_data):
    """
    Fits to the binned mass data. Using scipy.interpolate spline.
    
    Args:
    -------
    y_data: mass values, numpy array
    x_data: radius values, numoy array
    
    Returns:
    -------
    spline: a spline object
    """
    spline = UnivariateSpline(x_data, y_data, s=0)
    
    # Plotting the fitted spline for sanity check.
    """
    # Plot the original data
    plt.scatter(x_data, y_data, label='Data')
    # Plot the spline
    y_spline = spline(x_data)
    plt.plot(x_data,y_spline, label='Spline', color='red')
    plt.legend()
    plt.show()
    """
    return spline
   
 
def generating_r_distribution_for_binaries(spline):
    """
    Args:
    -------
    spline: a spline object
    
    Returns:
    -------
    star: array of float64; gives the location of the binaries generated from
         spline used as a pdf.
    """
    #subclass created to use random variate sampling
    class binary_distribution(rv_continuous):
        def __init__(self, name='binary', a=0., b=GALACTIC_RADIUS/8., res=BIN_NUMBER):
            # a and b give the limits of the galaxy
            
            super(binary_distribution, self).__init__(name=name,a=a,b=b)
            
            # defining the binary distribution as the spline fit
            self.binary_dist = spline
            
            self.x = np.linspace(a,b,res)
            self.y = self.binary_dist(self.x)
            
            # calculating the integral in the range of the galaxy
            self.norm = spline.integral(self.a,self.b)
            
        # overwriting the _pdf() method, while normalising the distribution
        def _pdf(self, x):
            return self.binary_dist(x)/self.norm
        
    # instance of the class created
    spline_dist = binary_distribution()
    
    # binary stars generated using the above written random variate sampler
    star = spline_dist.rvs(size=SAMPLE_SIZE)
    
    # Plotting the distribution for sanity check
    """
    x_values = np.linspace(0.,GALACTIC_RADIUS/8., 100)
    pdf_values = spline_dist.pdf(x_values)
    plt.plot(x_values, pdf_values, label='Spline PDF')

    plt.legend()
    plt.show()
    """
    return star


def isotropic_direction():
    """
    Creates an isotropic direcetion for each binary.
    
    Returns:
    -------
    isotropic: 2D nunmpy array of size (SAMPLE_SIZE,3) with the isotropic kicks.
    """
    
    
    isotropic = np.random.normal(size=(SAMPLE_SIZE,3))
    isotropic /= np.sqrt(np.sum(isotropic**2, axis=1)[:,np.newaxis])
    
    return isotropic


def kick_magnitude_data(kick_magnitudes, isotropic):
    """
    Creates kicks form the list in the data file.
    
    Args:
    -------
    kick_magnitudes: 1D numpy array of length SAMPLE_SIZE
    
    Returns:
    -------
    final_kicks: 2D numpy array of dimension (SAMPLE_SIZE,3) 
    """    
    
    # I don't think this is right
    final_kicks_list = []
    
    for i in range(len(kick_magnitudes)):
        final_kicks_list.append(np.sqrt(kick_magnitudes[i]**2 * (isotropic[i,:]**2)) * np.sign(isotropic[i,:]))
        
    final_kicks = np.array(final_kicks_list)
    
    return final_kicks


def kick_magnitudes_custom(isotropic):
    """
    Scales the isotropic kicks with a magnitude.
    
    Args:
    -------
    kick_velocities: 1D numpy array of length 5, could be chosen different
    
    Returns:
    -------
    final_kicks: 2D numpy array of dimension (SAMPLE_SIZE,3) 
    """
    kick_velocities = KICK_VELOCITY_MAGNITUDES*np.array([1.,2.,3.,4.,5.])

    final_kicks_list = []

    for i in range(len(kick_velocities)):
        final_kicks_list.append(np.sqrt(kick_velocities[i]**2 * (isotropic**2)) * np.sign(isotropic))
    
    final_kicks = np.array(final_kicks_list)
    
    return final_kicks


def circular_velocity(binary):
    """
    Calculates the circular velocity of the binary.
    
    Args:
    -------
    binary: A radius at which the binary resides.
    
    Returns:
    -------
    circ_velocity: The circular velocity of the binary in internal units.
    """
    
    circ_velocity = vcirc(MWPotential2014,binary)
    
    return circ_velocity


def constructing_items(star):
    
    
    items = []
    
    for i, binary in enumerate(star):
        items.append([binary,0.,circular_velocity(binary)/220.,0.,0.,0.]) #[R,vR,vT,z,vz,phi]
    
    return items


def updating_items(final_kicks, items):
    
    items_updated = []
    for i, kick in enumerate(final_kicks):
        items_updated.append([items[i][0],items[i][1] + kick[0],items[i][2] + kick[1],0.,items[i][4] + kick[2],0.])
    
    return items_updated


def orbit_objects(items):
    
    orbits = []

    for i in range(SAMPLE_SIZE):
        orbits.append(Orbit(items[i][:]))
        
    return orbits


def time_step(final_time=HUBBLE_TIME,start_time=0):
    
    """
    if (final_time == 10**10) :
        integration_step = 1000
        
    elif ((final_time - start_time) < 10**7) :
        integration_step = 10
        
    else:
        integration_step = int((final_time - start_time)/10**6)
    """
    integration_step=100
    ts = np.linspace(start_time,final_time,integration_step)*u.yr
    
    return ts


def time_one_revolution(star):
    
    time_1_rev = []

    for i in range(SAMPLE_SIZE): 
       time_1_rev.append((2*np.pi*star[i]*8.*3.085678*10**16)/circular_velocity(star[i])/31557600/10**9)
       
    time_1_revolution = np.array(time_1_rev)
    
    return time_1_revolution


def orbit_integraion_for_custom_kicks(star, potential):
    
    
    items1 = constructing_items(star)
    orbits1 = orbit_objects(items1)
    
    integrated_data1 = []
    
    time_one_rev = time_one_revolution(star)
    
    for i, orbit in enumerate(orbits1):
        ts1 = np.linspace(0.,time_one_rev[i],100)
        orbit.integrate(ts1*u.Gyr, potential)
        integrated_data1.append(orbit.getOrbit())
        
    # Doing the kick
    final_kicks = kick_magnitudes_custom(isotropic_direction()) 
    
    kick50 = final_kicks[0]
    
    # Evolving for Hubble time
    items2 = updating_items(kick50, items1)
    orbits2 = Orbit(items2)
    
    integrated_data2 = []
    
    # Integration step 2 is 1000
    ts2 = np.linspace(0.,14.,1000)
    
   
    orbits2.integrate(ts2*u.Gyr, potential)
    
    for i, orbit in enumerate(orbits2):
        integrated_data2.append(orbit.getOrbit())
       
    # Integration step 2 is 1000
    #for i in range(SAMPLE_SIZE):
        
        
    return integrated_data1, integrated_data2


def orbit_integration(star):
    
    # Integration untill the 1st supernova:
    items1 = constructing_items(star)
    orbits1 = orbit_objects(items1)
    
    integrated_data1 = []
    
    for i,orbit in enumerate(orbits1):
        orbit.integrate(time_step(final_time=data[i,3]),MWPotential2014)
        integrated_data1.append(orbit.getOrbit())
        
    # Doing 1st kick
    kick1 = kick_magnitude(data[:,5],isotropic_direction())
    
    items2 = updating_items(kick1, items1)
    orbits2 = orbit_objects(items2)
    
    integrated_data2 = []
    
    for i,orbit in enumerate(orbits2):
        orbit.integrate(time_step(final_time=data[i,4],start_time=data[i,3]),MWPotential2014)
        integrated_data2.append(orbit.getOrbit())
    
    # Doing 2nd kick
    kick2 = kick_magnitude(data[:,6],isotropic_direction())
    
    items3 = updating_items(kick2, items2)
    orbits3 = orbit_objects(items3)
    
    integrated_data3 = []
    
    for i,orbit in enumerate(orbits3):
        orbit.integrate(time_step(start_time=data[i,4]),MWPotential2014)
        integrated_data3.append(orbit.getOrbit())
        
    
    return integrated_data1, integrated_data2, integrated_data3


"""def plotting(data1,data2,data3):
    
    data1np = np.array(data1)
    data2np = np.array(data2)
    data3np = np.array(data3)    
    
    all_data_np = np.concatenate((data1np, data2np, data3np), axis=1)
    
    all_data_np[:,:,0] *= 8.
    all_data_np[:,:,3] *= 8.
    all_data_np[:,:,1] *= 220.
    all_data_np[:,:,2] *= 220.
    all_data_np[:,:,4] *= 220.
    
    figure = plt.figure(figsize=(26,26))
    xy_plot = figure.add_subplot(221)
    j = 13
    for i in [j]:#range(NUMBER_OF_BINARIES):
        x = np.empty((100,300))
        y = np.empty((100,300))
        x[i,:] = (all_data_np[i,:,0] * np.cos(all_data_np[i,:,5]))
        y[i,:] = (all_data_np[i,:,0] * np.sin(all_data_np[i,:,5]))
        xy_plot.plot(x[i,:], y[i,:], color='green')
        xy_plot.plot(x[i,:100],y[i,:100], color='red', linewidth= 5)
        xy_plot.scatter(x[i,100],y[i,100], color='black', marker='o', s=500, zorder=2)
        xy_plot.scatter(x[i,-1],y[i,-1], color='orange', marker='o', s=500, zorder=2)

    xy_plot.set_title('Orbits in the x-y plane', fontsize = 40)
    xy_plot.set_xlabel('x (kpc)', fontsize=35)
    xy_plot.set_ylabel('y (kpc)', fontsize=35)
    xy_plot.tick_params(axis='both', labelsize=30)
    
    figure.show()
    return all_data_np"""
"""
Next up: Orbit integeration and data recording.
    map: map(function,arr)
    orbits = list(map(...))
    https://docs.python.org/3/library/concurrent.futures.html
"""


def main_for_custom():
    
    mass, radius = binning_potential(galactic_potential())
    star = generating_r_distribution_for_binaries(fitting_to_binned_mass(radius, mass))
    a ,b = orbit_integraion_for_custom_kicks(star, galactic_potential())
    
    return 0

potential = galactic_potential()
m, radius = binning_potential(potential)
star = generating_r_distribution_for_binaries(fitting_to_binned_mass(radius, m))
a, b = orbit_integraion_for_custom_kicks(star, MWPotential2014)
##%%
"""
if file_validation():
    data = sample_data(read_input())
    mass_bin, r = binning_potential(galactic_potential())
    star = generating_r_distribution_for_binaries(fitting_to_binned_mass(r, mass_bin))
    #kick = kick_magnitude(data[:,5],isotropic_direction())
    #items = constructing_items(star)
    #items1 = updating_items(kick, items)
    #orbits = orbit_objects(items)
    idata1,idata2,idata3 = orbit_integration(star)
"""
#%%
INTEGRATION_STEP_1 = 100
INTEGRATION_STEP_2 = 1000

a_np = np.array(a)
b_np = np.array(b)

all_data_np = np.concatenate((a, b), axis=1)


all_data_np[:,:,0] *= 8.
all_data_np[:,:,3] *= 8.
all_data_np[:,:,1] *= 220.
all_data_np[:,:,2] *= 220.
all_data_np[:,:,4] *= 220.


#find max r for each orbit
#

for i in range(SAMPLE_SIZE):
    R = np.empty((SAMPLE_SIZE,INTEGRATION_STEP_1 + INTEGRATION_STEP_2)) 
    R[i,:] = (all_data_np[i,:,0])
    


#np.savetxt("data_50.csv", all_data_np, delimiter=",") #3D array not 2D think if this is needed 








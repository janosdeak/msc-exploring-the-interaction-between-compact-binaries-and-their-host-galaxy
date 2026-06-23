# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:33:53 2024

@author: Kecsi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, MovingObjectPotential, HomogeneousSpherePotential, mass, evaluatePotentials, evaluateRforces
from astropy.coordinates import SkyCoord, CylindricalRepresentation
import astropy.units as u



# Convert Galactic coordinates to cylindrical coordinates
sky_coord = SkyCoord(ra=80.89*u.deg, dec=-69.76*u.deg, distance=50000*u.pc)
cylindrical_coord = sky_coord.represent_as(CylindricalRepresentation)

# Get cylindrical coordinates (rho, phi, z)
rho = cylindrical_coord.rho
phi0 = cylindrical_coord.phi.radian
z0 = cylindrical_coord.z
rho = rho.value
z0 = z0.value

converted = np.empty((1,3))

converted[0,0] = rho
converted[0,1] = z0
converted[0,2] = phi0

LMC_orbit = Orbit([rho/8000., 0., 31.5/220., z0/8000., 0., phi0]) # [R,vR,vT,z,vz,phi] 

LMC_local_pot = HomogeneousSpherePotential(amp=1,R=0.625,normalize=13.8) #6.25 = 5/8

LMC_global_pot = MovingObjectPotential(LMC_orbit, pot= LMC_local_pot)

# mlmc = LMC_local_pot.mass(R=1)
# mmw = [] 
# for i in range(3):
#     mmw.append(MWPotential2014[i].mass(R=0.75*10**4))
    
final_MW = MWPotential2014 + LMC_global_pot

bound_limit = evaluatePotentials(MWPotential2014, R=3.375, z=0.)


from scipy.constants import hbar, pi, electron_mass, alpha, c, elementary_charge
import scipy.constants

a0 = scipy.constants.physical_constants['Bohr radius'][0] # m

# [s]
us = 1e-6
ms = 1e-3

# [1/s]
kHz = 1e3
MHz = 1e6  
GHz = 1e9

# [W]
mW = 1e-3 

# [m]
nm = 1e-9
um = 1e-6
mm = 1e-3

# [K]
mK = 1e-3
uK = 1e-6

# constants
h = 2*pi*hbar


def get_atomic_pol_unit():
    """get atomic polarizability unit

    Args:
        hatree energy unit [J]

    returns:
        atomic polarizability unit [C^2 m^2 J^-1]
    """
    hartree_energy = electron_mass*c**2*alpha**2 # J
    atomic_polarizability_unit = elementary_charge**2*a0**2/hartree_energy
    return atomic_polarizability_unit


atomic_pol_unit = get_atomic_pol_unit()
    
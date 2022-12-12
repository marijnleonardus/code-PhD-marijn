# author: Marijn Venderbosch
# november 2022

"""functions on converting to an approriate set of units"""

from scipy.constants import hbar, electron_mass, fine_structure, c, e
     
        
def get_bohr_radius():
    """get Both radius from physical constants"""
    
    a0 = hbar / electron_mass / c / fine_structure
    return a0


def get_hartree_energy_unit():
    """get Hartree energy unit in terms of physical constants and Bohr radius"""

    bohr_radius = get_bohr_radius()
    energy_h = hbar**2 / electron_mass / bohr_radius**2
    return energy_h


def get_atomic_unit():
    """get atomic unit in terms of Bohr radius and Hartree energy unit"""

    hartree_energy_unit = get_hartree_energy_unit()
    bohr_radius = get_bohr_radius()
    
    au = e**2 * bohr_radius**2 / hartree_energy_unit
    return au

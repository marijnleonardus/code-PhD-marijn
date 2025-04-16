# author: Marijn Venderbosch
# 2023

from arc import Strontium88, PairStateInteractions
from functions.conversion_functions import get_atomic_pol_unit
from scipy.constants import c, epsilon_0


def calculate_c6_coefficients(n, l, j, mj):
    """calculate C6 coefficients using ARC library
    Assumes the quantum numbers are identical for the 2 atoms
    
    parameters
    -------------
    n: integer
        principal quantum number 
    l: integer:
        angular momentum quantum number
    j: integer: 
        total angular momentum quantum number
    mj: integer:
        secondary total angular momentum quantum number
    
    returns
    ----------------
    c6: float
        van der waals interaction coefficient
        
    example
    -------------
    So for (61s5s) 3P0 mj=0 state of Sr88:
    - 61, 0, 1, 0, 1
    """
    
    calc = PairStateInteractions(Strontium88(),
                                 n, l, j,
                                 n, l, j,
                                 mj, mj,
                                 s=1)
    theta = 0
    phi = 0
    deltaN = 6
    deltaE = 30e9 # in Hz
    
    c6, eigenvectors = calc.getC6perturbatively(theta, phi, 
                                                deltaN, deltaE,
                                                degeneratePerturbation=True)
    # getC6perturbatively returns the C6 coefficients
    # expressed in units of GHz mum^6.
    
    # Conversion to atomic units:
    #c6 = c6 / 1.4458e-19
    
    # These results should still be divided by n^{11}
    # to be plotted as in Fig. 2(c).
    return c6

 
def ac_stark_shift_polarizability(alpha, intensity):
    """    
    compute AC Stark shift given intensity and polarizability alpha
    
    inputs;
    - polarability alpha
    - intensity [W/m^2]
    
    returns:
    - AC stark shift"""
    
    # atomic polarizability
    atomic_polarizability = get_atomic_pol_unit()
    
    # stark shift
    shark_shift = alpha * atomic_polarizability / (2 * c * epsilon_0) * intensity
    return shark_shift



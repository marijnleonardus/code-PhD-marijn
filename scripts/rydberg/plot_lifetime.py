# author: Marijn Venderbosch
# January 2023

from arc import Strontium88
import matplotlib.pyplot as plt
import numpy as np

Sr88=Strontium88()
plt.style.use('default')


def lifetime(n):
    """
    computes rydberg lifetime (black body, sponatneous emission)
    from Madav Mohan paper fig 6.

    Parameters
    ----------
    n : integer
        printipal quantum number.

    Returns
    -------
    lifetime : float
        rydberg lifetime in [s].

    """
    
    # fit parameters from his paper
    A = 18.84311101
    B = 875.31756526
    
    # quantum defect 
    delta = Sr88.getQuantumDefect(n, 0, 1, s=1)
    
    # compute lifetime rydberg state in s
    lifetime = (A*(n-delta)**(-2)+B*(n-delta)**(-3))**(-1)*1e-6
    return lifetime


n = np.linspace(30, 100, 71)
lifetimes = lifetime(n)

fig, ax = plt.subplots()
ax.plot(n, lifetimes/1e-6)
ax.set_xlabel('$n$')
ax.set_ylabel('$ns5s {}^3S_1$ lifetime [$\mu$s]')

        

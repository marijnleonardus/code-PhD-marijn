# author: Robert de Keijzer, Marijn Venderbosch
# january 2023

import numpy as np
from scipy.constants import electron_mass, c, alpha, hbar, alpha
from scipy.constants import epsilon_0 as eps0
from scipy.constants import elementary_charge as e0
import scipy.constants



# %% variables

a0 = scipy.constants.physical_constants['Bohr radius'][0] # m
hartree_energy = electron_mass * c**2 * alpha**2 # J
t = c**2 * alpha**2


# %% functions


def rate_to_rdme(Aki, J, En1, En2):
    #from Einstein coefficient to Radial density matrix element
    #See Eq. 27 from DOI: 10.1119/1.12937
    
    rdme = np.sqrt(3 * np.pi * eps0 * hbar * c**3 / (a0**2 * e0* 2 *np.abs(En1 - En2)**3) * (2 * J + 1) * Aki)
    return rdme


def rdme_to_rate(rdme, J, En1, En2):
    # looks like Energies are actually angular frequnecies (so not multiplied by hbar)
    
    rate = np.abs(En1 - En2)**3 * e0**2 / (3 * np.pi * eps0 * hbar* c**3 *(2 * J + 1)) * (a0 * rdme)**2
    return rate


def rabi_freq_to_rate(intensity, rabi_freq, omega21):
    # given intensity and transition frequency
    
    rate = hbar *  omega21**3 * rabi_freq**2 / (6 * np.pi * c**2 * intensity)
    return rate


def wavelength_to_freq(wavelength):
    # wavelength in m to frequency in Hz

    frequency = 2 * np.pi * c / wavelength
    return frequency


def energy_to_wavelength(transition_energy):
    # transition energy in joule, wavelength in m
    
    wavelength = 2 * np.pi * hbar * c / transition_energy
    return wavelength


def rdme_to_rabi(rdme, intensity):
    """inputs
        intensity [W]: gaussian beam intensity
        RDME [atomic units]: radial dipole matrix element <g|r|e>
    
    returns:
        Rabi frequency [Hz]
    """
    
    rabi = (rdme * e0 * a0) / hbar * np.sqrt(2 / (c * eps0)) * np.sqrt(intensity) 
    return rabi


def gaussian_beam_intensity(beam_waist, power):
    """inputs:
        beam_waist [m]
        power [W]
        
    returns:
        I0 (max intensity) [W/m^2]"""
    
    I0 = 2 * power / (np.pi * beam_waist**2)
    return I0



# #magnetic field in Gauss to magnetic field in Tesla
# def gauss_to_tesla(B):
#     return B*10**(-4)

# #magnetic field in Tesla to magnetic field in Gauss
# def tesla_to_gauss(B):
#     return B*10**4

# # Polarizability in SI units (Cm^2/V) to Polarizability in atomic units (a.u.)
# def cm2Vmin1_to_AU(pol):
#     return (e0**2*a0**2/Eh)**(-1)*pol

# # Polarizability in SI units (Cm^2/V) to Polarizability in atomic units (a.u.)
# def AU_to_cm2Vmin1(pol):
#     return (e0**2*a0**2/Eh)*pol

# #Wavenumber in cm^(-1) to Wavelength in m
# def cmmin1_to_wavelength(k):
#     return 1/(10**2*k)

# #Wavenumber in cm^(-1) to Frequency in Hz
# def cmmin1_to_freq(k):
#     return 2*np.pi*c*(10**2*k)

# #Frequency in Hz to Wavenumber in cm^(-1)
# def freq_to_cmmin1(w):
#     return w/(2*np.pi*c*(10**2))

# #Wavelength in m to Wavenumber in cm^(-1)
# def wavelength_to_cmmin1(w):
#     return 1/(10**2*w)

# #Wavenumber in cm^(-1) to Energy in J
# def cmmin1_to_joule(k):
#     return hbar*2*np.pi*c*(10**2)*k

# #Wavenumber in cm^(-1) to Energy in Hartree
# def cmmin1_to_hartree(k):
#     return 2*np.pi*(10**2)*alpha*a0*k

# #Wavelength in m to Energy in Joule 
# def wavelength_to_joule(w):
#     return hbar*2*np.pi*c/w

# #intensity in Watt/m^2 to intensity in miliWatt/cm^2
# def wm2_to_mwcm2(I):
#     return I*10**3*10**(-4)

# #energy in Joule to Wavelength in m
# def joule_to_wavelength(E):
#     return hbar*2*np.pi*c/E

# # Energy in Joule to Frequency in Hz
# def joule_to_frequency(E):
#     return E/hbar

# #frequency in Hz to energy in Joule
# def frequency_to_joule(f):
#     return f*hbar

# #frequency in Hz to wavelength in m
# def frequency_to_wavelength(w):
#     return 2*np.pi*c/w

# #energy in electronvolt to energy in joule
# def ev_to_joule(E):
#     return E*e0

# #energy in ev to frequency in Hz
# def ev_to_frequency(E):
#     return E*e0/hbar

# #energy in electronvolt to wavelength in m
# def ev_to_wavelength(E):
#     return hbar*2*np.pi*c/(E*e0)

# #wavenumber in cm^(-1) to wavelength in m
# def wavenumber_to_wavelength(k):
#     return 10**(-2)/k

# #wavenumber in cm^(-1) to frequency in Hz
# def wavenumber_to_frequency(k):
#     return 2*np.pi*c*k*10**(2)

# #wavelength in m to energy in hartree
# def wavelength_to_hartree(wl):
#     return alpha*a0*2*np.pi/wl

# #energy in hartree to wavelength in m
# def hartree_to_wavelength(E):
#     return alpha*a0*2*np.pi/E

# #energy in hartree to frequency in Hz
# def hartree_to_freq(E):
#     return c*E/(alpha*a0*10**9)

# #frequency in Hz to energy in Hartree
# def freq_to_hartree(E):
#     return E*(alpha*a0*10**9)/c
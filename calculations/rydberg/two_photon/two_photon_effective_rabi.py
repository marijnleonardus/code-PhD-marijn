# author: Marijn Venderbosch
# 2023
"""
computes effective rabi frequency from two-photon transition as well as
off-resonant scattering rate from intermediate state |e> using adiabatic elimination

three level system
 |g> = 1S0
 |e> = 3P1
 |r> = 61sns 3S1
"""

import numpy as np
from classes.conversion_class import Conversion
from classes.optics_class import Optics

# %% variables

pi=np.pi
wavelength_ge = 689e-9  # [m]
NA = 0.5  # numerical aperture objective

#  linewidth |g> to |3p1> or red MOT line
linewidth_ge= 2*pi*7.4e3  # [Hz]

# power split off from red MOT lasersent to microscope objective
power_ge = 0.06e-6  # [W] 

# beam parameters
beamwaist_ge = Optics.gaussian_beam_diffraction_limit(wavelength_ge, NA)  # [m]
intensity_ge = Optics.gaussian_beam_intensity(beamwaist_ge, power_ge)  # [W/m^2]

# rabi frequency that is obtained 
freq_ge = Conversion.wavelength_to_freq(wavelength_ge)  # [Hz]
rabi_ge = Conversion.rate_to_rabi(intensity_ge, linewidth_ge, freq_ge)  # [Hz]

# |e> to |r>
rabi_er = 2*pi*10e6  # [Hz]
detuning_r = 0  # [Hz] detuning from rydberg state

# detuning 3P1 (max. accesible with AOM)
detuning_e = 2*pi*100e6# Hz


# %% functions

def two_photon_effective_detuning(rabi_ge, rabi_er, detuning_e, detuning_r):
    """
    Parameters
    ----------
    rabi_ge : float
        rabi frequency from |g> to |e>.
    rabi_er : float
        rabi frequency from |e> to rydberg state |r>.
    detuning_e : float
        detuning from intermediate state |e>.
    detuning_r : float
        detuning from rydberg state |r>.

    Returns
    -------
    Effective detuning treating it as a two level system.
    """
    eff_detuning = detuning_r+rabi_ge**2/(4*detuning_e)+rabi_er**2/(4*detuning_e)
    return eff_detuning


def two_photon_effective_rabi(rabi_ge, rabi_er, detuning_e):
    """
    Parameters
    ----------
    rabi_ge : float
       rabi frequency from |g> to |e>.
    rabi_er : float
        rabi frequency from |e> to rydberg state |r>.
    detuning_e : float
        detuning from intermediate state |e>.

    Returns
    -------
    Effective rabi frequency from |g> to |r> skipping intermediate state
    """
    eff_rabi = rabi_ge*rabi_er/(2* detuning_e)
    return eff_rabi

# %% calculations

effective_rabi = two_photon_effective_rabi(rabi_ge, rabi_er, detuning_e)
effective_detuning = two_photon_effective_detuning(rabi_ge, rabi_er, detuning_e, detuning_r)

print('rabi_ge is 2pi x ' +str(np.round(rabi_ge/(2*pi*1e6), 0)) + ' MHz')
print('rabi_er is 2pi x ' +str(np.round(rabi_er/(2*pi*1e6), 0)) + ' MHz')
print('two photon rabi freq is ' +str(np.round(effective_rabi/(2*pi*1e6), 1)) + ' MHz')


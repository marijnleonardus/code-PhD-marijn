# author: Marijn Venderbosch
# January 2023

import numpy as np
from scipy.constants import electron_mass, c, hbar, alpha
from scipy.constants import epsilon_0 as eps0
from scipy.constants import elementary_charge as e0
import scipy.constants


# %% variables

a0 = scipy.constants.physical_constants['Bohr radius'][0] # m
hartree_energy = electron_mass * c**2 * alpha**2 # J
t = c**2 * alpha**2


# %% functions

class Conversion:

    def rate_to_rdme(Aki, J, En1, En2):
        """
        from Einstein coefficient to Radial density matrix element
        See Eq. 27 from DOI: 10.1119/1.12937
        
        inputs
        - Aki: einstein coefficient [Hz]
        - J: quantum number
        - En1 and En2 [Hz] transition energy/h
        
        returns:
        - radial dipole matrix element in atomic units [a0*e]
        """
        
        rdme = np.sqrt(3 * np.pi * eps0 * hbar * c**3 / (a0**2 * e0* 2 *np.abs(En1 - En2)**3) * (2 * J + 1) * Aki)
        return rdme


    def rdme_to_rate(rdme, J, En1, En2):
        """
        From Radial density matrix element to Einstein coefficient
        
        inputs
        - rdme: matrix element in atomic units
        - J: quantum number
        - En1 and En2 [Hz] transition energy/h
        
        returns:
        - Einstein coefficient in Hz
        """
        
        rate = np.abs(En1 - En2)**3 * e0**2 / (3 * np.pi * eps0 * hbar* c**3 *(2 * J + 1)) * (a0 * rdme)**2
        return rate
    
    
    def rate_to_rabi(intensity, linewidth, omega21):
        """
        inputs:
        - intensity [W/m^2]
        - linewidth/Einstein coefficient [Hz]
        - Transition angular freq omega21 [rad/s]
        
        returns:
        - Rabi frequency [Hz]
        """
        rabi_square = 6 * np.pi * c**2 * intensity * linewidth / (hbar * omega21**3)
        rabi = rabi_square**(0.5)
        return rabi
        

    def rabi_freq_to_rate(intensity, rabi_freq, omega21):
        """
        inputs:
        - intensity [W/m^2]
        - Rabi freq [Hz]
        - Transition angular freq omega21 [rad/s]
        
        returns:
        - einstein coefficient [Hz]
        """
        
        rate = hbar *  omega21**3 * rabi_freq**2 / (6 * np.pi * c**2 * intensity)
        return rate
    

    def intensity_to_electric_field(intensity):
        """
        inputs:
        - intensity [W/m^2]
        
        returns:
        - electric field strength [V/m]
        """
        electric_field_square = 2 * intensity / (c * eps0)
        electric_field = np.sqrt(electric_field_square)
        return electric_field


    def wavelength_to_freq(wavelength):
        """
        inputs:
        - wavelength [m]
        
        returns:
        - angular freq [rad/s]
        """

        frequency = 2 * np.pi * c / wavelength
        return frequency


    def energy_to_wavelength(transition_energy):
        """
        inputs:
        - transition energy [J]
        - wavelength [m]
        
        returns:
        - transition wavelength
        """
        
        wavelength = 2 * np.pi * hbar * c / transition_energy
        return wavelength


    def rdme_to_rabi(rdme, intensity, j_e):
        """
        computes Rabi frequency given RDME (radial dipole matrix element)

        Parameters
        ----------
        rdme : float
            radial dipole matrix elment in [atomic units].
        intensity : float
            laser intensity in [W/m^2].
        j_e : integer
            quantum number J for rydberg state.

        Returns
        -------
        rabi : float
            rabi freq. in [Hz].

        """
    
        rabi = (rdme*e0*a0)/hbar*np.sqrt(2*intensity/(eps0*c*(2*j_e+1)))
        return rabi


    def gaussian_beam_intensity(beam_waist, power):
        """
        inputs:
        - beam_waist [m]
        - power [W]
            
        returns:
        - I0 (max intensity) [W/m^2]
        """
        
        I0 = 2 * power / (np.pi * beam_waist**2)
        return I0


    def cylindrical_gaussian_beam(waist_x, waist_y, power):
        """
        gaussian beam but waist in x and y are not the same (cylindrical)
        
        inputs:
        - beam_waist in x and y directoins [m]
        - power [W]
            
        returns:
        - I0 (max intensity) [W/m^2]
        """
            
        I0 = 2 * power / (np.pi * waist_x * waist_y)
        return I0


    def saturation_intensity(lifetime, wavelength):
        """
        inputs:
        - wavelength in m
        - excited state lifetime tau in s

        returns:
        - saturation intensity
        """
        isat = np.pi * (hbar * 2 * np.pi) * c / (3 * lifetime * wavelength**3)
        return isat


    def compute_rabi_freq(rdme, electric_field):
        """
        inputs:
        - rdme in atomic units [a_0 * e]
        - intensity in [W / m^2]
        
        returns:
        - rabi frequency
        """
        # convert to SI units from atomic
        rdme_se = abs(rdme *a0 * e0) # coulom * m

        # compute rabi frequency
        omega = electric_field / hbar * rdme_se
        return omega
    

    def compute_ac_stark_shift(rabi_freq, detuning):
        """
        inputs:
        - rabi frequency in [Hz]
        - detuning in [Hz]
        
        returns:
        - AC Stark shift in [Hz]
        """
        stark_shift = rabi_freq**2 / (4 *detuning)
        return stark_shift
    
    def dc_stark_shift(polarizability, electric_field):
        """
        see paper Mohan 2022 for Sr88 datda

        Parameters
        ----------
        polarizability : float
            in units of [MHz cm^2 V^-2].
        electric_field : float
            in units of [V/cm].

        Returns
        -------
        None.

        """
        dc_stark = 1/2*polarizability*electric_field**2
        return dc_stark

    def get_atomic_pol_unit():
        """
        inputs:
            - bohr radius [m]
            - hatree energy unit [J]
        returns:
            -atomic polarizability unit
        """
        
        au = e0**2 * a0**2 / hartree_energy
        return au

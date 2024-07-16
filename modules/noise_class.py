# author: Marijn Venderbosch
# 2023

import numpy as np
import scipy.integrate
from arc import Strontium88
from scipy.constants import pi 

Sr88 = Strontium88()


class GateErrors:
    @staticmethod
    def atomic_motion_error(k, variance, mass, rabifreq):
        """
        Parameters
        ----------
        k : float
            2*pi/lambda, wavenumber in [m^-1].
        variance : float
            given doppler distribution, this is the variance in [m^2/s^2]
        mass : float
            mass of atom in [kg].
        rabifreq : float
            rabi freq of |g> to |r> oscillatoins in [Hz].
    
        Returns
        -------
        error :float
            1-fidelity, error in 2 qubit gate fidelity.
    
        """
        error=5*k**2*variance/(4*mass**2*rabifreq**2)
        return error
    
    @staticmethod
    def imperfect_blockade_error(rabifreq, interaction):
        """
        Parameters
        ----------
        rabifreq : float
            rabi frequency of in [Hz] |g> to |r> in [Hz] so not enhanced Rabi frequencyh.
        interaction : float
            interaction energy of dipole-dipole in [Hz].
    
        Returns
        -------
        error: float
            error in fidelity as a result from blockade not being perfect.
        """
        error = 0.5*rabifreq**2/interaction**2
        return error
    

class IntensityNoise:
    def shot_to_shot_fidelity_error(sigma):
        """
        compute entangling error as a result of intensity noise
        assuming only shot-to-shot noise

        Parameters
        ----------
        sigma : float
            spread in intensity/I0.

        Returns
        -------
        error : float
            fidelity error 1-F.
        """
        fidelity=0.5*(1 + np.exp(-0.5*pi**2*sigma**2))
        error=1 - fidelity
        return error
        
    
    # Functions from thesis Kale, 2020 to compute for given PSD of noise fidelity error
    # commented for now because we assume only shot to shot noise
    
    # def window_function_0(w, t):
    #     """
    #     windows function that selects relevant of intensity spectrum as a function of time
    #     see thesis Anant Kale, p. 22

    #     Parameters
    #     ----------
    #     w : array
    #         frequency, independent variable (omega) [Hz].
    #     t : float
    #         time in [s].

    #     Returns
    #     -------
    #     window: array
    #         no unit

    #     """
        
    #     window = np.sin(w*t/2)**2/(w*t/2)**2
    #     return window
    
    # def product_function_0(rabi, omega, intensityspectrum):
    #     """
    #     computes product of window function with intensity noise spectrum
    #     so that this function can be integrated at time t1=pi/rabi

    #     Parameters
    #     ----------
    #     rabi : float
    #         rabi frequency in [Hz].
    #     omega : float
    #         frequency in [Hz].
    #     intensityspectrum: array
    #         intensity noise spectrum as function of fourier freq

    #     Returns
    #     -------
    #     None.

    #     """
    #     t1 = pi/rabi
    #     window1t = IntensityNoise.window_function_0(omega, t1)
        
    #     product = intensityspectrum*window1t
    #     return product
    
    # def compute_fidelity_error(rabi, omega, noise_spectrum):
    #     """
    #     computes fidelity error as result of phase noise 
    #     see thesis Anant Kale eq. 4.24 and eq. 4.57
    
    #     Parameters
    #     ----------
    #     rabi: float
    #         rabi freq in [Hz]
    #     omega: array
    #         array of independent variables (frequency)
    #     noise_spectrum: array
    #         intensity noise spectrum as function of omega        
    
    #     Returns
    #     -------
    #     fidelity_error : float
    #         error in pi pulse fidelity.
    
    #     """
    #     # enhanced rabi freq because of blockade
    #     rabi=np.sqrt(2)*rabi
        
    #     # compute product function for specific rabi frequency and frequencies array
    #     product1t = IntensityNoise.product_function_0(rabi, omega, noise_spectrum)
        
    #     # integrate this spectrum
    #     integral = scipy.integrate.simpson(product1t, omega)
        
    #     fidelity_error = pi**2/2*1/(2*pi)*integral
    #     return fidelity_error

class PhaseNoise:
    @staticmethod
    def window_function_1(r, w, t):
        """
        windows function that selects relevant of frequency spectrum as a function of time
        see thesis Anant Kale, p. 23
    
        Parameters
        ----------
        r : float
            rabi frequency in [Hz].
        w : float
            frequency, independent variable (omega) [Hz].
        t : float
            time in [s].
    
        Returns
        -------
        window : array
            spectral window function that selects parts of spectrum.
    
        """
        numerator = 3*r**2+w**2 + (r**2 - w**2)*np.cos(2*r*t) - 4*r**2*np.cos(r*t)*np.cos(w*t) - 4*r*w*np.sin(r*t)*np.sin(w*t)

        # compute denominator and fraction
        denominator = (r**2 - w**2)**2*t**2
        window = numerator/denominator
        return window 
    
    @staticmethod
    def product_function_1(rabi, omega, noise_spectrum):
        """
        computes product of window function with frequency noise spectrum
        so that this function can be integrated at time t1=pi/rabi
    
        Parameters
        ----------
        rabi : float
            rabi frequency in [Hz].
        omega : float
            frequency in [Hz].
        noise_spectrum: array
            noise spectrum as function of fourier freq
    
        Returns
        -------
        product : array
            window function times noise spectrum
    
        """
        t1 = pi/rabi
        window1t = PhaseNoise.window_function_1(rabi, omega, t1)
        
        product = window1t*noise_spectrum
        return product
    
    @staticmethod
    def compute_fidelity_error(rabi, omega, noise_spectrum):
        """
        computes fidelity error as result of phase noise 
        see thesis Anant Kale eq. 4.24 and eq. 4.57
    
        Parameters
        ----------
        rabi : float
            rabi frequency in [Hz].
        omega: array
            array of independent variables (frequency)
    
        Returns
        -------
        fidelity_error : float
            error in pi pulse fidelity.
    
        """
        # enchance rabi freq by factor sqrt(2) because of blockade
        rabi = np.sqrt(2)*rabi
        
        # compute product function for specific rabi frequency and frequencies array
        product1t = PhaseNoise.product_function_1(rabi, omega, noise_spectrum)
        
        # integrate this spectrum
        integral = scipy.integrate.simpson(product1t, omega)
        
        fidelity_error = pi**2/(8*rabi**2)*integral
        return fidelity_error
    
class Lifetime:
    @staticmethod
    def rydberg_state(n):
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
        lifetime = (A*(n - delta)**(-2) + B*(n - delta)**(-3))**(-1)*1e-6
        return lifetime

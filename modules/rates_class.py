# author: Marijn Venderbosch
# january 2023

from arc import Strontium88
import scipy
import os 

# add modules to path
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../modules'))
sys.path.append(modules_dir)

# load user defined module
from conversion_class import Conversion

bohr_radius = scipy.constants.physical_constants['Bohr radius'][0]
Sr88 = Strontium88()


class LightAtomInteraction:
    
    def scattering_rate_sat(detuning, linewidth, s0):
        """off-resonant scattering rate given saturation parameter s_0
        
        input: 
        - saturation paramter s0 
        - detuning [rad/s]
        - linewidth [rad/s]
        
        returns:
        - scattering rate [rad/s]     
        """
        
        rate = .5*s0*linewidth/(1 + s0 + (2*detuning/linewidth)**2)
        return rate
    
    def scattering_rate_power(linewidth, detuning, wavelength, beam_waist, power):
        """
        computes off-resonant scattering rate for a transition with linewidth
        assumes gaussian beam geometry
        
        inputs:
        - linewidth [rad/s]
        - detuning [rad/s]
        - wavelength [m]
        - beam_waist [m]
        - laser power [W]
        
        returns:
        - off resonant scattering rate in [rad/s]
        """
        
        # lifetime excited state
        lifetime = 1/linewidth
        
        # saturation intensity, intensity, and s0 parameter
        sat_intensity = Conversion.saturation_intensity(lifetime, wavelength)
        intensity = Conversion.gaussian_beam_intensity(beam_waist, power)
        s0 = intensity/sat_intensity
        
        # off-resonant scattering rate computed from scattering rate formula
        off_resonant_rate = LightAtomInteraction.scattering_rate_sat(detuning, linewidth, s0)
        return off_resonant_rate

    def sr88_rdme_value_au(n):
        """
        computes RDME (radial dipole matrix element) for Sr88 in atomic units
        fucnction is obtained from fitting experimental data of 3P1-r
        which is equivalent for 3P0-r up to a Glebsch-Gordan coefficient

        Parameters
        ----------
        n : integer
            principal quantun number.

        Returns
        -------
        rdme : float
            radial dipole matrix element in [a .

        """
        # get quantum defect for 3S1 state (l=0, j=1, s=1)
        defect = Sr88.getQuantumDefect(n, 0, 1, s=1)
        
        # get RDME in atomic units
        rdme_au = 1.728*(n - defect)**(-1.5)
        
        return rdme_au
# author: Marijn Venderbosch
# january 2023

from functions.conversion_functions import saturation_intensity, gaussian_beam_intensity


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
        
        rate = .5 * s0 * linewidth / (1 + s0 + (2 * detuning / linewidth)**2)
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
        lifetime = 1 / linewidth
        
        # saturation intensity
        sat_intensity = saturation_intensity(lifetime, wavelength)

        # intensity 
        intensity = gaussian_beam_intensity(beam_waist, power)
        
        # saturation parameter
        s0 = intensity / sat_intensity
        
        # off-resonant scattering rate computed from scattering rate formula
        off_resonant_rate = LightAtomInteraction.scattering_rate_sat(detuning, linewidth, s0)
        return off_resonant_rate


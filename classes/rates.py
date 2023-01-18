# author: Marijn Venderbosch
# january 2023

from functions.conversion_functions import saturation_intensity, gaussian_beam_intensity


class LightAtomInteraction:
    
    def off_resonant_scattering(linewidth, detuning, wavelength, beam_waist, power):
        
        # lifetime excited state
        lifetime = 1 / linewidth
        
        # saturation intensity
        sat_intensity = saturation_intensity(lifetime, wavelength)
        print(sat_intensity)
        # intensity 
        intensity = gaussian_beam_intensity(beam_waist, power)
        
        # saturation parameter
        s0 = intensity / sat_intensity
        
        # scattering rate
        rate = .5 * s0 * linewidth / (1 + s0 + (2 * detuning / linewidth)**2)
        return rate
    
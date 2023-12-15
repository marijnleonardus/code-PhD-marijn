# author: Marijn Venderbosch
# July 2023

from scipy.constants import pi


class GaussianBeam:

    def __init__(self, power, waist):
        self.power = power
        self.waist = waist

    def get_intensity(self):
        """Intensity of Gaussian beam"""
            
        intensity = 2*self.power/pi/self.waist**2
        return intensity
    
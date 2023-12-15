# author: Marijn Venderbosch
# july 2023

from scipy.constants import epsilon_0, c


class AtomicCalculations:

    def __init__(self, atomic_unit):
        self.au = atomic_unit

    def ac_stark_shift(self, polarizability, intensity):
        """
        returns AC stark shift

        Args:
            polarizability (float): in atomic units
            intensity (float): unit W/m^2

        Returns:
            AC stark shift: trap depth in J
        """
              
        shark_shift = polarizability*self.au/(2*c*epsilon_0)*intensity
        return shark_shift
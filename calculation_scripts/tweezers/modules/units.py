# author: Marijn Venderbosch
# november 2022

from scipy.constants import hbar, electron_mass, fine_structure, c, e


class UnitConversion:
     
    def get_bohr_radius(self):
        """bohr radius

        Returns:
            a0: bohr radius in [m], float
        """
        
        a0 = hbar/electron_mass/c/fine_structure
        return a0

    def get_hartree_unit(self, bohr_radius):
        """get hartree fock energy unit

        Returns:
            energy_h: hartree fock energy unit in [J]
        """

        energy_h = hbar**2/electron_mass/bohr_radius**2
        return energy_h

    def get_atomic_unit(self, bohr_radius, hartree_unit):
        """get atomic unit

        Returns:
            au: atomic unit
        """
        
        au = e**2*bohr_radius**2/hartree_unit
        return au

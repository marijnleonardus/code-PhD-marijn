# author: Marijn Venderbosch
# january 2023

import numpy as np
from modules.optics_class import Optics


# %% variables
wavelength_rydberg = 317e-9
numerical_aperture = 0.5
atom_spacing = 5e-6


# %% 
diffraction_limited_waist = Optics.gaussian_beam_diffraction_limit(wavelength_rydberg,
                                                                   numerical_aperture)

rel_int = Optics.gaussian_beam_radial(atom_spacing, diffraction_limited_waist)

rabi_rel_int = np.sqrt(rel_int)
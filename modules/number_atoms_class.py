# author: Marijn Venderbosch
# December 2022

import numpy as np
from scipy.constants import pi

class NumberAtoms:
    """collection of funcions to compute number of atoms and density of atoms
    from pictures"""
    
    @staticmethod
    def atom_number_from_image(fit_parameters_x, fit_parameters_y, max_counts, gain,
        exp_time, constant_counts, loss_constant, linewidth, lens_dist, lens_rad):
        """assuming MOT is described by gaussian in x and y,
        the integral over the region of interest is given by
        2 * pi * amplitude * sigma_x * sigma_y
        """
    
        # obtain sigma x,y from the fit 
        sigma_horiz = fit_parameters_x[3]
        sigma_vert = fit_parameters_y[3]
        
        # amplitude of fits: x and y don´t have to be exactly the same, 
        # theefore take the average
        amplitude_weighted = 0.5*(fit_parameters_x[0] + fit_parameters_y[0])*max_counts
    
        # obtain average number of counts
        counts_roi = 2*pi*amplitude_weighted*sigma_horiz*sigma_vert
        # formula 5.1 from Msc thesis Venderbosch
    
        # prefactor 2 because atoms occupy 1/2 of the time the excited state,
        # assuming saturation intensity
        atomic_part_prefactor = 2/loss_constant/linewidth
    
        # fraction of light collected by lens
        collection_part = 4*pi*lens_dist**2/lens_rad**2
    
        # pixel counts
        camera_counts = counts_roi/(exp_time*gain*1/constant_counts)
    
        atoms_mot = atomic_part_prefactor*collection_part*camera_counts
        return atoms_mot

    @staticmethod
    def atomic_density_from_atom_number(atom_number, sigma_horiz, sigma_vert):
        
        """given atom number and sigma (1/e size MOT) computes atomic density
        sigma can be different in horizontal and vertical 
        assumes cloud has sigma_horiz in x,y and sigma_vert in z direction"""
        
        # convert to m
        """unsure why 1e5 here?? (note july '24 while cleaning up)"""
        sigma_x_m = sigma_horiz/1e5
        sigma_y_m = sigma_vert/1e5
        
        # first compute SI units
        atom_density_si_units = atom_number/(2*pi)**(3/2)/(sigma_x_m**2*sigma_y_m)
        
        # convert to per cubic cm
        atom_density_cm3 = atom_density_si_units/(1e2)**3
        return atom_density_cm3

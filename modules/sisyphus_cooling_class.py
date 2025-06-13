# Helper functions for Sisyphus cooling simulation

import numpy as np
from qutip import *

# add local modules
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
if modules_dir not in sys.path:
    sys.path.append(modules_dir)
from atoms_tweezer_class import AtomicMotion


class SisyphusCooling:
    def __init__(self, N_max, N_i, mass, lamb, wg, thetas, d_theta):
        # Initialize parameters
        self.mass = mass
        self.lamb = lamb
        self.wg = wg
        self.thetas = thetas
        self.d_theta = d_theta

        # construct operators
        self.psi0 = tensor(basis(2, 0), fock(N_max, N_i))
        self.a = tensor(qeye(2), destroy(N_max))
        self.emit = tensor(Qobj([[0, 1], [0, 0]]), qeye(N_max))
        self.absorb = self.emit.dag()
        self.project_e = tensor(projection(2, 1, 1), qeye(N_max))
        self.project_g = tensor(projection(2, 0, 0), qeye(N_max))
        self.number_op = self.a.dag()*self.a

    def get_operators(self):
        operators = [self.psi0, self.project_e, self.project_g, self.number_op]
        return operators

    def calculate_rad_pattern(self, theta, d_theta, me=1):
        """Radiation pattern integration weight"""
        if me == 0:
            pre_factor = 3/4*np.sin(theta)**2
        else:
            pre_factor = 3/8*(1 + np.cos(theta)**2)
        return pre_factor*np.sin(theta)*d_theta

    def calculate_H(self, linewidth, rabi_f, wg, we, detuning):
        """Compute sparse Hamiltonian (nondimensional units)"""

        # compute Lamb-Dicke parameter
        eta = AtomicMotion().lamb_dicke_parameter(self.mass, self.lamb, self.wg)

        # nondimensionalize
        freqs = np.array([linewidth, rabi_f, wg, we, detuning])/rabi_f
        _, rabi_nd, wg_nd, we_nd, det_nd = freqs

        H = (
            wg_nd*(self.a.dag()*self.a + 0.5)*(self.project_g + self.project_e)
            + (we_nd**2 - wg_nd**2)/(4*wg_nd)*(self.a + self.a.dag())**2*self.project_e
            + rabi_nd/2*(
                (1j*eta*(self.a + self.a.dag())).expm()*self.absorb
                + (-1j*eta*(self.a + self.a.dag())).expm()*self.emit
            ) - det_nd*self.project_e)
        return H

    def calculate_c_ops(self, linewidth, rabi_f, thetas, d_theta):
        """Precompute sparse collapse operators for given angles"""
        
        linewidth_nd = linewidth/rabi_f
        c_ops = []
        for theta in thetas:
            rate = linewidth_nd*self.calculate_rad_pattern(theta, d_theta)
            kick = (-1j*(self.a + self.a.dag())*np.cos(theta)).expm()
            c_ops.append(np.sqrt(rate)*kick*self.emit)
        return c_ops
    
    def solve_master_equation(self, arguments):
        "to be used in for loop, "
        "this time we have to compute c_ops each time since they vary with rabi frequency, which "
        "could be the argument in the for loop"

        [linewidth, rabi_f, we, detuning, times_rabi] = arguments
        psi0, project_e, _, number_op = self.get_operators()
        H = self.calculate_H(linewidth, rabi_f, self.wg, we, detuning)
        c_ops = self.calculate_c_ops(linewidth, rabi_f, self.thetas, self.d_theta)
        sol = mesolve(H, psi0, times_rabi, c_ops, e_ops=[project_e, number_op], options={"store_states": False})
        return sol

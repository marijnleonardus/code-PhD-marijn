import matplotlib.pyplot as plt
import numpy as np
import pylcp


class Plotting:    
    
    def __init__(self, length_scale, time_scale):
        self.length_scale = length_scale
        self.time_scale = time_scale
    
    def force_profile_3d(self, equation):
        """plot MOT force in x,y,z"""
        
        #length_mm = 
        length_mm = 1e3*self.length_scale  # mm, for plotting rescaling

        x = y = z = np.linspace(-0.3, 0.3, 1001)/length_mm
        r = np.array([x, y, z])
        v = np.zeros((3,) + z.shape)

        # generate force profile for all positions
        equation.generate_force_profile(r, v, name='force3_d')

        # plot force profile
        fig0, ax0 = plt.subplots(1, 1, figsize=(4, 3))
        ax0.plot(x*length_mm, equation.profile['force3_d'].F[0], label=r'$F_x$')
        ax0.plot(y*length_mm, equation.profile['force3_d'].F[1], label=r'$F_y$')
        ax0.plot(z*length_mm, equation.profile['force3_d'].F[2], label=r'$F_z$')

        # cosmetics
        ax0.set_xlabel('$r_i$ (mm)')
        ax0.set_ylabel('$F_i/(\hbar k \Gamma)$')
        ax0.legend()
        ax0.axvline(0, color='black')
        ax0.axhline(0, color='black')
        plt.show()


    def trajectory_single_atom(self, equation, t_max):
        """plot trajectory of one atom only to check everything went well"""

        if isinstance(equation, pylcp.rateeq):
            equation.set_initial_pop(np.array([1., 0., 0., 0.]))
        equation.set_initial_position(np.array([0., 0., 0.]))

        # simulate for 10% of the time
        equation.evolve_motion([0, t_max*0.1], random_recoil=True, progress_bar=True, max_step=1.)

        # plot test solution
        fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.75))
        ax[0].plot(equation.sol.t*self.time_scale, equation.sol.r.T*(1e6*self.length_scale))
        ax[1].plot(equation.sol.t*self.time_scale, equation.sol.v.T)
        ax[0].set_ylabel('$r$ ($\mu$m)')
        ax[0].set_xlabel('$t$ (s)')
        ax[1].set_ylabel('$v/(\Gamma/k)$')
        ax[1].set_xlabel('$t$ (s)')
        fig.subplots_adjust(left=0.08, wspace=0.22)
        plt.show()

        # delete test solution
        if hasattr(equation, 'sol'):
            del equation.sol

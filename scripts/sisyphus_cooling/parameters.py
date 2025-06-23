"""parameter to be used during Sisyphus cooling simulations

The reason they are here, is that there are separate scripts, but they should use the same parameters.
"""	

from scipy.constants import pi, proton_mass
import numpy as np

# add local modules
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
if modules_dir not in sys.path:
    sys.path.append(modules_dir)
utils_dir = os.path.abspath(os.path.join(script_dir, '../../utils'))
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from units import kHz, nm

# physical parameters
linewidth = 2*pi*7.4*kHz
rabi_f = 2*pi*36*kHz
alpha_e = 355 # polarizability, a.u.
alpha_g = 286 # polarizability, a.u.
wg = 2*pi*86*kHz
we = np.sqrt(alpha_e/alpha_g)*wg
detuning = -2*pi*160*kHz  # detuning from the excited state
mass = 87.9*proton_mass
lamb = 689*nm

# simulation parameters
# motional levels
N_max = 20      # motional levels
N_i = 12           # initial Fock level

# emission angle discretization
N_theta = 30
thetas = np.linspace(0, pi, N_theta)
d_theta = thetas[1] - thetas[0]

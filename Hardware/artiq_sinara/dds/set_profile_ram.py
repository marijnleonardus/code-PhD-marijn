# author: Marijn Venderbosch
# October 2022


"""script used to figure out what gets called when the function `set_profile_ram` is called
from https://m-labs.hk/artiq/manual-beta/core_drivers_reference.html?highlight=ad9910#module-artiq.coredevice.ad9910"""

"""Set the RAM profile settings.

:param start: Profile start address in RAM.
:param end: Profile end address in RAM (last address).
:param step: Profile time step in units of t_DDS, typically 4 ns
            (default: 1).
:param profile: Profile index (0 to 7) (default: 0).
:param nodwell_high: No-dwell high bit (default: 0,
            see AD9910 documentation).
:param zero_crossing: Zero crossing bit (default: 0,
            see AD9910 documentation).
:param mode: Profile RAM mode (:const:`RAM_MODE_DIRECTSWITCH`,
:const:`RAM_MODE_RAMPUP`, :const:`RAM_MODE_BIDIR_RAMP`,
:const:`RAM_MODE_CONT_BIDIR_RAMP`, or
:const:`RAM_MODE_CONT_RAMPUP`, default:
:const:`RAM_MODE_RAMPUP`)
"""

# %% imports

import numpy as np
import matplotlib.pyplot as plt

# %% functions


def bitsequence_sent(start, end, step, profile, nodwell_high, mode, zero_crossing):
    # high and low
    hi = (step << 8) | (end >> 2)
    lo = ((end << 30) | (start << 14) | (nodwell_high << 5) |(zero_crossing << 3) | mode)
    
    # bitsequence
    bit_sequence = (bin(hi).replace("0b", "")[-32:] + bin(lo).replace("0b", "")[-32:])[::-1]
    #print("Binary string: " + str(bit_sequence))
    
    # timestep
    timestep_binary = bit_sequence[40:56][::-1]
    
    
    return int(timestep_binary, 2)

# %% apply function and plot

step_size = int(1e6)
step_size_matrix = np.linspace(1, step_size, step_size-1)
 
M_matrix = np.zeros(step_size, dtype=int)

for i in range(1, step_size):
    M_number = bitsequence_sent(start = 0, 
                                end = 1024 - 1,
                                step = i,
                                profile = 0,
                                nodwell_high = 0,
                                mode = 4, 
                                zero_crossing = 0)
    M_matrix[i] = M_number

plt.plot(M_matrix)



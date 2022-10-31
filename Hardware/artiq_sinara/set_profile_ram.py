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

# %% variables
start = 0
end = 1024-1
step = int(12)
profile = 0
mode = 4
nodwell_high = 0
zero_crossing=0

# high and low
hi = (step << 8) | (end >> 2)
lo = ((end << 30) | (start << 14) | (nodwell_high << 5) |(zero_crossing << 3) | mode)

# bitsequence
bit_sequence = (bin(hi).replace("0b", "")[-32:] + bin(lo).replace("0b", "")[-32:])[::-1]
print("Binary string: " + str(bit_sequence))

# timestep
timestep_binary = bit_sequence[40:56][::-1]
print("Step (Binary): " + str(timestep_binary))
print("Step value (M in datasheet): "+ str(int(timestep_binary,2)))
print("Step value (ns): "+str(4 * int(timestep_binary, 2)))
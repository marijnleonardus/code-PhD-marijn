# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 09:34:37 2022

@author: farubidium
"""

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
start = 2
end = 1024-1
step = 0000000000001100
profile = 0
mode = 1
nodwell_high = 0
zero_crossing=0
hi = (step << 8) | (end >> 2)
lo = ((end << 30) | (start << 14) | (nodwell_high << 5) |(zero_crossing << 3) | mode)
#print(bin(hi).replace("0b", ""))
#print(bin(lo).replace("0b", ""))
s = (bin(hi).replace("0b", "")[-32:]+bin(lo).replace("0b", "")[-32:])[::-1]

stepBin = s[40:56][::-1]
print("Binary string: " + str(s))
print("Step (Binary): "+str(stepBin))
print("Step value (M in datasheet): "+str(int(stepBin,2)))
print("Step value (ns): "+str(4*int(stepBin,2)/1E9*1E9))
#print(len(bin(step).replace("0b", "")))
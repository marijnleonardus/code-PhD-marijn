import numpy as np
from scipy.constants import c


# %% variables
freq_RR = 250e6
freq_CEO = 35e6

f1 = c / 922e-9
f2 = c / 813e-9
f3 = c / 707e-9
f4 = c / 698e-9
f5 = c / 689e-9
f6 = c / 688e-9
f7 = c / 679e-9

offset = (f1 / freq_RR - int(f1 / freq_RR)) / int(f1 / freq_RR) * freq_RR + freq_CEO / (f1 / freq_RR)

def beat(fn, n):
    return ((freq_RR + offset + freq_RR**2 / f1 * n) * int(fn / freq_RR) - fn) / freq_RR - np.round(((freq_RR + offset + freq_RR**2 / f1 * n) * int(fn / freq_RR) - fn) / freq_RR,
                                                                                                                  decimals = 0)


for i in range(1000000):
    list_plus_beats = [np.abs(beat(f1, i)),
                       np.abs(beat(f2, i)),
                       np.abs(beat(f3, i)),
                       np.abs(beat(f4, i)),
                       np.abs(beat(f5, i)),
                       np.abs(beat(f6, i)),
                       np.abs(beat(f7, i))
                       ]
    
    list_minus_beats = [np.abs(beat(f1, -i)),
                        np.abs(beat(f2, -i)),
                        np.abs(beat(f3, -i)),
                        np.abs(beat(f4, -i)), 
                        np.abs(beat(f5, -i)),
                        np.abs(beat(f6, -i)),
                        np.abs(beat(f7, -i))
                        ]
    
    if max(list_plus_beats) < 0.2 or max(list_minus_beats) < 0.2: 
        if min(list_plus_beats) > 0.1 or min(list_minus_beats) > 0.1: 
            print("frepuency offset = " + str(offset + freq_RR**2 / f1 * i))
        else: 
            continue
    else: 
        continue

print(beat(f1,-i))
print(beat(f2,-i))
print(beat(f3,-i))
print(beat(f4,-i))
print(beat(f5,-i))
print(beat(f6,-i))
print(beat(f7,-i))
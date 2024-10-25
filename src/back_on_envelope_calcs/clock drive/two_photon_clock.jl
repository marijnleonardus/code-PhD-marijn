# script for estimating a 2 photon rabi frequency when driving the 2 photon 
# optical clock transition of Sr-88

# assume 100 mW of clock laser power, 10 um x 10 um area
power = 100e-3  # W
waist = 10e-6  # m
area = π*waist^2  # m^2

# calculate intensity
intensity = 2*power/area
println(intensity)

# coupling strength from https://arxiv.org/pdf/1405.1330
coupling_strength = 8.8e-6 # Hz/(W/m^2)

# calculate rabi frequency
# the rabi freq. for this transition scales proportional to intensity
rabi_freq = intensity*coupling_strength
println(rabi_freq)
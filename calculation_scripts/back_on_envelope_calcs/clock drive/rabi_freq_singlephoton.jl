# see madjarov thesis eq. 3.4
using PhysicalConstants.CODATA2018

# constants
ħ =  ReducedPlanckConstant
gs = 2
gl = 1
μ = BohrMagneton
c = SpeedOfLightInVacuum

# variables tunable experiment
B = 0.016 # [T]
waist = 200e-6 # [m]
power = 40e-3  # [W]

# variables fixed
delta01 = 2*π*5.6e12  # [Hz]
gamma3p1 = 2*π*7.4e3  # [Hz]
lambda = 698.4e-9  # [m]

# compute intensity
intensity = 2*power/(π*waist^2)

# compute rabi freq. in [Hz], see eq 3.2 in thesis Madjarov
# i take abs of μc, otherwise you get a negative nr
μc = sqrt(2/3)*(gl - gs)*μ   # [J/T]
rabi_freq = abs(μc)*B/(delta01*ħ) * sqrt(3*gamma3p1*lambda^3*intensity/(4*π^2*ħ*c))
println(rabi_freq/(2*π))

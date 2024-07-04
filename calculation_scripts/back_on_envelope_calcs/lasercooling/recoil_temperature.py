from scipy.constants import Boltzmann, hbar, proton_mass,pi

lamb = 461e-9
m=88*proton_mass
k=2*pi/lamb

recoil_energy = hbar**2*k**2/m
recoil_temperature = recoil_energy/Boltzmann

print(recoil_energy)
print(recoil_temperature)


# coding: utf-8

# In[1]:


import numpy as np
#import pandas as pd 
#import tables as tb
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.constants import c, h, k, m_p

#get_ipython().magic('matplotlib inline')


# In[16]:



conversion_factor = 1.64878e-41 #multiply the polarizability by this number to get results in J
eta0_freespace = 376.7 #Ohms
amu = 1.66e-27 #multiply the atomic mass in amu units by this number to get the answer in kg
gEarth = 9.81 #acceleration due to gravity

pi = np.pi



def I_circular_gaussian(r,power,w0): #this is at z = 0
    prefactor = 2*power/(np.pi*w0**2)
    result = prefactor*np.exp((-2*r**2)/(w0**2))
    return result

def potential_min(power,w0,polarizability):
    result_J = -0.5*eta0_freespace*I_circular_gaussian(0,power,w0)*conversion_factor*polarizability
    result_Hz = result_J/const.h
    result_K = result_J/const.Boltzmann
    return result_K

def potential_radial(r,power,w0,polarizability):
    result_J = -0.5*I_circular_gaussian(r,power,w0)*eta0_freespace*conversion_factor*polarizability
    result_K = result_J/const.Boltzmann
    return result_K


def potential_z(z,power,w0,lam_light,polarizability,atomic_mass_amu):
    evolution = 1 + ((lam_light*z)/(np.pi*w0**2))**2 
    result_J = -0.5*eta0_freespace*conversion_factor*polarizability*(2*power)/(np.pi*evolution*w0**2) + atomic_mass_amu*amu*gEarth*z
    result_K = result_J/const.Boltzmann
    return result_K

def oscillator_freq_r(power,w0,polarizability,atomic_mass_amu):
    result_omega_sq = (4*polarizability*conversion_factor*eta0_freespace*power)/(atomic_mass_amu*amu*np.pi*w0**4)
    result_Hz = np.sqrt(result_omega_sq)/(2*np.pi)
    return result_Hz

def oscillator_freq_z(power,w0,lam_light,polarizability,atomic_mass_amu):
    result_omega_sq =(polarizability*conversion_factor*eta0_freespace*power*lam_light**2)/(atomic_mass_amu*amu*np.power(np.pi,3)*np.power(w0,6))
    result_Hz = np.sqrt(result_omega_sq)/(2*np.pi)
    return result_Hz


# ### Calculations
# This is alll commented out past calculations for different papers and situtations scroll down for the most recent and relavent calculations
# ##### Trap depth and freqs for a given beam geometry as a function of power

# In[5]:


# beam_w0 = .82e-6 #NOTE: These are "radii" not "diameters"
# atomic_mass = 88
# lam_light = 515.2e-9
# laserpowers = np.linspace(0.5,12,100)*1e-3
# trap_depths = np.array([potential_min(power,beam_w0,alpha1S0_532) for power in laserpowers])
# trap_freq_r = np.array([oscillator_freq_r(power,beam_w0,alpha1S0_532,atomic_mass) for power in laserpowers])
# trap_freq_z = np.array([oscillator_freq_z(power,beam_w0,lam_light,alpha1S0_532,atomic_mass) for power in laserpowers])
# plt.plot(laserpowers*1e3,1e6*trap_depths)
# plt.title("w0 = {:.2f} microns".format(beam_w0*1e6))
# plt.xlabel("Laser beam power [mW]")
# plt.ylabel("Trap depth [microK]")
# plt.show()
# plt.plot(laserpowers*1e3,trap_freq_r*1e-3)
# plt.title("w0 = {:.2f} microns".format(beam_w0*1e6))
# plt.xlabel("Laser beam power [mW]")
# plt.ylabel("Trap freq r [kHz]")
# plt.show()
# plt.plot(laserpowers*1e3,trap_freq_z*1e-3)
# plt.title("w0 = {:.2f} microns".format(beam_w0*1e6))
# plt.xlabel("Laser beam power [mW]")
# plt.ylabel("Trap freq z [kHz]")
# plt.show()


# ###   Let us see what happens if we add gravity to the potential in the z-direction

# In[6]:


# beam_w0 = .8e-6
# z_positions = np.linspace(-1e-4,1e-4,301)
# laserpower = 25e-3
# laserpowerManuel=6e-3
# #1S0 for 532
# plt.plot(z_positions*1e6,potential_z(z_positions,laserpower,beam_w0,lam_light,alpha1S0_532,atomic_mass)*1e6)

# plt.title("Potential 1S0 including gravity, power {:.3f} mW".format(laserpower*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [micro K]")
# plt.show()

#3p1 potentials for 532
#plt.plot(z_positions*1e6,potential_z(z_positions,laserpower,beam_w0,lam_light,alpha3P1_zero_lin_532,atomic_mass)*1e6)
#plt.title("Potential 3P1, linear polarization,mj=0. including gravity, power {:.3f} mW".format(laserpower*1e3))
#plt.xlabel("Position [microns]")
#plt.ylabel("Energy [micro K]")
#plt.show() Commented out because we measured circular polarizaton


# plt.plot(z_positions*1e6,potential_z(z_positions,laserpower,beam_w0,lam_light,alpha3P1_zero_right_532,atomic_mass)*1e6)
# plt.title("Potential 3P1, right/left polarization,mj=0. including gravity, power {:.3f} mW".format(laserpower*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [micro K]")
# plt.show()


#plt.plot(z_positions*1e6,potential_z(z_positions,laserpower,beam_w0,lam_light,alpha3P1_min1_lin_532,atomic_mass)*1e6)
#plt.title("Potential 3P1, linear polarization,mj=-1. including gravity, power {:.3f} mW".format(laserpower*1e3))
#plt.xlabel("Position [microns]")
#plt.ylabel("Energy [micro K]")
#plt.show() Commented out because we measured circular polarizaton


# plt.plot(z_positions*1e6,potential_z(z_positions,laserpower,beam_w0,lam_light,alpha3P1_min1_right_532,atomic_mass)*1e6)
# plt.title("Potential 3P1, right polarization,mj=-1. including gravity, power {:.3f} mW".format(laserpower*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [micro K]")
# plt.show()

# plt.plot(z_positions*1e6,potential_z(z_positions,laserpower,beam_w0,lam_light,alpha3P1_min1_left_532,atomic_mass)*1e6)
# plt.title("Potential 3P1, left polarization,mj=-1. including gravity, power {:.3f} mW".format(laserpower*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [micro K]")
# plt.show()



#1S0 potential Manuel 520
# plt.plot(z_positions*1e6,potential_z(z_positions,laserpowerManuel,.5e-6,520e-9,878,atomic_mass)*1e6)
# plt.title("Potential 1S0 Manuel at 520 including gravity, power {:.3f} mW".format(laserpowerManuel*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [micro K]")
# plt.show()

# #3p1 potentials Manuel 520



# plt.plot(z_positions*1e6,potential_z(z_positions,laserpowerManuel,.5e-6,520e-9,877.3,atomic_mass)*1e6)
# plt.title("Potential 3P1 Manuel, 520 ,mj=0. including gravity, power {:.3f} mW".format(laserpowerManuel*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [micro K]")
# plt.show()

# plt.plot(z_positions*1e6,potential_z(z_positions,laserpowerManuel,.5e-6,520e-9,573.4,atomic_mass)*1e6)
# plt.title("Potential 3P1,Manuel,520,mj=+/-1. including gravity, power {:.3f} mW".format(laserpowerManuel*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [micro K]")
# plt.show()





# #1S0 potential Manuel 515

# plt.plot(z_positions*1e6,potential_z(z_positions,.1e-3,1.2e-6,515.2e-9,941.8,atomic_mass)*1e6)
# plt.title("Potential 1S0 at 515.2 including gravity 20 mW .75 waist".format(laserpowerManuel*1e3))
# plt.xlabel("Position [microns]")
# plt.xlim(-40,40)
# plt.ylabel("Energy [micro K]")
# plt.show()



# #3P1 potential Manuel 515

# plt.plot(z_positions*1e6,potential_z(z_positions,laserpowerManuel,.5e-6,515.2e-9,990.4,atomic_mass)*1e6)
# plt.title("Potential 3P1 Manuel, 515.2 ,mj=0. including gravity, power {:.3f} mW".format(laserpowerManuel*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [micro K]")
# plt.show()

# plt.plot(z_positions*1e6,potential_z(z_positions,laserpowerManuel,.5e-6,515.2e-9,658.9,atomic_mass)*1e6)
# plt.title("Potential 3P1,Manuel,515.2,mj=+/-1. including gravity, power {:.3f} mW".format(laserpowerManuel*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [micro K]")
# plt.show()









# ho_freq_r = oscillator_freq_r(laserpower,beam_w0,alpha1S0_532,atomic_mass)
# ho_freq_z = oscillator_freq_z(laserpower,beam_w0,lam_light,alpha1S0_532,atomic_mass)

# ground_state_shift = np.min(potential_z(z_positions,laserpower,beam_w0,lam_light,alpha1S0_532,atomic_mass)*const.Boltzmann/const.h)+            0.5*const.h*(2*ho_freq_r+ho_freq_z)
# print("The ac Stark shift of the HO ground state is in Hz: ",ground_state_shift)

# tdepth = (potential_z(-30e-6,laserpower,beam_w0,lam_light,alpha3P1_zero_left_532,atomic_mass))-(potential_z(0,laserpower,beam_w0,lam_light,alpha3P1_zero_left_532,atomic_mass))
# print("Trap depth as measured from z={}um to z={}um: {} uK = {} MHz".format(-30,0,tdepth*1e6,tdepth*const.Boltzmann/const.h/1e6))



# # Stark shift of 1S0 with respect to vacuum 

# """
# 25/7/18: Ivo wrote this small piece of code to save the potential 
# as a .dat file for plotting purposes. Feel free to delete or edit
# at will. 

# try:
#     f = open("potentialsave.dat","w+")
# except PermissionError:
#     sys.exit("Could not open the file.")
# with f:
#     f.write("x\tf(x)\n")
#     f.write("# Potential including gravity for 0.5W. Units: um, uK \n")
#     for i in range(301):
#         f.write("{}\t{}\n".format(z_positions[i]*1e6,potential_z(z_positions[i],laserpower,beam_w0,lam_light,alpha1S0_532,atomic_mass)*1e6))
#     print("File saved.")
# """


# # In[7]:



# #This is the calculations that match Manuels findings. From what I understand, the large negatvie shift
# #corresponds to the Phi_b and Phi_c for linear light like we have.
# #The alpha=991.2 would be the state used for sideband cool Phi_a



# beam_w0_Manuel = .5e-6
# r_positions = np.linspace(-1e-5,1e-5,301)
# laserpowerManuel=2.7e-3



# #1S0 potential Manuel 520
# #plt.plot(z_positions*1e6,potential_radial(r_positions,laserpowerManuel,.5e-6,878)*1e6)
# #plt.title("Potential 1S0 Manuel at 520 in radial plane, power {:.3f} mW".format(laserpowerManuel*1e3))
# #plt.xlabel("Position [microns]")
# #plt.ylabel("Energy [micro K]")
# #plt.show()





# #3p1 potentials Manuel 520



# #plt.plot(r_positions*1e6,potential_radial(r_positions,laserpowerManuel,.5e-6,877.3)*1e6)
# #plt.title("Potential 3P1 Manuel, 520 ,mj=0. in radial direction, power {:.3f} mW".format(laserpowerManuel*1e3))
# #plt.xlabel("Position [microns]")
# #plt.ylabel("Energy [micro K]")
# #plt.show()

# #plt.plot(r_positions*1e6,potential_radial(r_positions,laserpowerManuel,.5e-6,573.4)*1e6)
# #plt.title("Potential 3P1,Manuel,520,mj=+/-1. in radial plane, power {:.3f} mW".format(laserpowerManuel*1e3))
# #plt.xlabel("Position [microns]")
# #plt.ylabel("Energy [micro K]")
# #plt.show()




# #1S0 potential Manuel 515

# plt.plot(r_positions*1e6,potential_radial(r_positions,laserpowerManuel,beam_w0_Manuel,941.8)*1e6)
# plt.title("Potential 1S0 Manuel at 515.2 in radial plane, power {:.3f} mW".format(laserpowerManuel*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [micro K]")
# plt.show()



# #3P1 potential Manuel 515

# plt.plot(r_positions*1e6,potential_radial(r_positions,laserpowerManuel,beam_w0_Manuel,990.4)*1e6)
# plt.title("Potential 3P1 Manuel, 515.2 ,mj=0. in radial plane, power {:.3f} mW".format(laserpowerManuel*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [micro K]")
# plt.show()

# plt.plot(r_positions*1e6,potential_radial(r_positions,laserpowerManuel,beam_w0_Manuel,658.9)*1e6)
# plt.title("Potential 3P1,Manuel,515.2,mj=+/-1. in radial plane, power {:.3f} mW".format(laserpowerManuel*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [micro K]")
# plt.show()

# plt.plot(r_positions*1e6,(potential_radial(r_positions,laserpowerManuel,beam_w0_Manuel,990.4)-potential_radial(r_positions,laserpowerManuel,beam_w0_Manuel,941.8))*1e6*20.837)
# plt.title("differential shift 1S0- 3P1, linear polarization,mj=0 in radial plane, power {:.3f} mW".format(laserpowerManuel*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [kHz]")
# plt.show()
# print("difference at center of trap [kHz][3P1mj=0-1S0]=")
# print((potential_min(laserpowerManuel,beam_w0_Manuel,990.4)-potential_min(laserpowerManuel,beam_w0_Manuel,941.8))*1e6*20.837)

# plt.plot(r_positions*1e6,(potential_radial(r_positions,laserpowerManuel,beam_w0_Manuel,658.9)-potential_radial(r_positions,laserpowerManuel,beam_w0_Manuel,941.8))*1e6*20.837)
# plt.title("differential shift 1S0- 3P1, linear polarization,mj=1 in radial plane, power {:.3f} mW".format(laserpowerManuel*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [kHz]")
# plt.show()
# print("difference at center of trap [kHz][3P1mj=1-1S0]=")
# print((potential_min(laserpowerManuel,beam_w0_Manuel,658.9)-potential_min(laserpowerManuel,beam_w0_Manuel,941.8))*1e6*20.837)

# print("FOR 532 nm TWEEZER USING DURHAM EXPERIMENTAL ALPHA VALUES")
# plt.plot(r_positions*1e6,potential_radial(r_positions,laserpower,beam_w0,alpha1S0_532)*1e6*20.837)
# plt.plot(r_positions*1e6,potential_radial(r_positions,laserpower,beam_w0,alpha3P1_zero_Durham)*1e6*20.837)
# plt.title("Potentials for 1S0 & 3P1, linear polarization,mj=0 in radial plane, power {:.3f} mW".format(laserpower*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [kHz]")
# plt.show()
# print("difference at center of trap [kHz][3P1mj=0-1S0]=")
# print((potential_min(laserpower,beam_w0,alpha3P1_zero_Durham)-potential_min(laserpower,beam_w0,alpha1S0_532))*1e6*20.837)
# plt.plot(r_positions*1e6,(potential_radial(r_positions,laserpower,beam_w0,alpha3P1_zero_Durham)-potential_radial(r_positions,laserpower,beam_w0,alpha1S0_532))*1e6*20.837)
# plt.title("differential shift 1S0- 3P1, linear polarization,mj=0 in radial plane, power {:.3f} mW".format(laserpower*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [kHz]")
# plt.show()

# plt.plot(r_positions*1e6,potential_radial(r_positions,laserpower,beam_w0,alpha1S0_532)*1e6*20.837)
# plt.plot(r_positions*1e6,potential_radial(r_positions,laserpower,beam_w0,alpha3P1_min1_Durham)*1e6*20.837)
# plt.title("Potentials for 1S0 & 3P1, linear polarization,mj=+/-1 in radial plane, power {:.3f} mW".format(laserpower*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [kHz]")
# plt.show()
# print("difference at center of trap [kHz][3P1mj=1-1S0]=")
# print((potential_min(laserpower,beam_w0,alpha3P1_min1_Durham)-potential_min(laserpower,beam_w0,alpha1S0_532))*1e6*20.837)
# plt.plot(r_positions*1e6,(potential_radial(r_positions,laserpower,beam_w0,alpha3P1_min1_Durham)-potential_radial(r_positions,laserpower,beam_w0,alpha1S0_532))*1e6*20.837)
# plt.title("differential shift 1S0- 3P1, linear polarization,mj=+/-1 in radial plane, power {:.3f} mW".format(laserpower*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [kHz]")
# plt.show()

# ### For our setup






# These values are from my polarizability calculations
beam_w0 = .8e-6
r_positions = np.linspace(-1e-5,1e-5,301)
laserpower = 20e-3
atomic_mass = 88.
pol1S0_785 = 291.38 
pol1S0_813 = 280.876 
pol1S0_532 = 756.107
# pol3P1_mj0_Pi_785 = 185.5
# pol3P1_mj1_Pi_785 = 443.4
# pol3P1_mj0_Pi_813 = 176.25
# pol3P1_mj1_Pi_813 = 378.694  These are the values I get from calculating using just NISt values
pol3P1_mj0_Pi_785 = 199.934
pol3P1_mj1_Pi_785 = 404.496
pol3P1_mj0_Pi_813 = 189.9
pol3P1_mj1_Pi_813 = 348.71
pol5d1D2_785 = -2000
pol3P1_mj0_Pi_532 = 694
pol3P1_mj1_Pi_532 = 430.625
# For mj=0 
# plt.plot(r_positions*1e6,potential_radial(r_positions,laserpower,beam_w0,pol1S0_785)*1e6)
# plt.plot(r_positions*1e6,potential_radial(r_positions,laserpower,beam_w0,pol3P1_mj0_Pi_785)*1e6)
# plt.title("Potentials for 1S0 & 3P1, linear polarization,mj=0 in radial plane, power {:.3f} mW".format(laserpower*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [microK]")
# plt.show()
print("For 785 nm")
print("Trap depth of 1S0(microK,kHz)=")
print(potential_min(laserpower,beam_w0,pol1S0_785)*1e6)
print(potential_min(laserpower,beam_w0,pol1S0_785)*1e6*20.837)
print("Trap depth of 3P1, mj=0 (microK,kHz)=")
print(potential_min(laserpower,beam_w0,pol3P1_mj0_Pi_785)*1e6)
print(potential_min(laserpower,beam_w0,pol3P1_mj0_Pi_785)*1e6*20.837)
print("Trap depth of 3P1, mj=1 (microK,kHz)=")
print(potential_min(laserpower,beam_w0,pol3P1_mj1_Pi_785)*1e6)
print(potential_min(laserpower,beam_w0,pol3P1_mj1_Pi_785)*1e6*20.837)

print("difference at center of trap (AC stark shift) [kHz][3P1mj=0-1S0]=")
print((potential_min(laserpower,beam_w0,pol3P1_mj0_Pi_785)-potential_min(laserpower,beam_w0,pol1S0_785))*1e6*20.837)
# plt.plot(r_positions*1e6,(potential_radial(r_positions,laserpower,beam_w0,pol3P1_mj0_Pi_785)-potential_radial(r_positions,laserpower,beam_w0,pol1S0_785))*1e6*20.837)
# plt.title("differential shift 1S0- 3P1, linear polarization,mj=0 in radial plane, power {:.3f} mW".format(laserpower*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [kHz]")
# plt.show()

# For mj=+/-1. For linear light these are degenerate

# plt.plot(r_positions*1e6,potential_radial(r_positions,laserpower,beam_w0,pol1S0_785)*1e6*20.837)
# plt.plot(r_positions*1e6,potential_radial(r_positions,laserpower,beam_w0,pol3P1_mj1_Pi_785)*1e6*20.837)
# plt.title("Potentials for 1S0 & 3P1, linear polarization,mj=+/-1 in radial plane, power {:.3f} mW".format(laserpower*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [kHz]")
# plt.show()
print("difference at center of trap (AC stark shift)  [kHz][3P1mj=1-1S0]=")
print((potential_min(laserpower,beam_w0,pol3P1_mj1_Pi_785)-potential_min(laserpower,beam_w0,pol1S0_785))*1e6*20.837)
# plt.plot(r_positions*1e6,(potential_radial(r_positions,laserpower,beam_w0,pol3P1_mj1_Pi_785)-potential_radial(r_positions,laserpower,beam_w0,pol1S0_785))*1e6*20.837)
# plt.title("differential shift 1S0- 3P1, linear polarization,mj=+/-1 in radial plane, power {:.3f} mW".format(laserpower*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [kHz]")
# plt.show()
print("Trap frequency  [kHz] between 1S0, radial:") 
print(oscillator_freq_r(laserpower,beam_w0,pol1S0_785,atomic_mass)/1000) 
print("Trap frequency  [kHz] between 1S0, axial:") 
print(oscillator_freq_z(laserpower,beam_w0,785e-9,pol1S0_785,atomic_mass)/1000)

# plt.plot(r_positions*1e6,potential_radial(r_positions,laserpower,beam_w0,pol5d1D2_785)*1e6)
# plt.title("differential shift 1S0- 3P1, linear polarization,mj=0 in radial plane, power {:.3f} mW".format(laserpower*1e3))
# plt.xlabel("Position [microns]")
# plt.ylabel("Energy [microk]")
# plt.show()





# ### We can also check out the trap frequency difference between the 1S0 state and some substates for 3P1 for the purposes of sideband cooling

# In[9]:



print("Trap frequency difference [kHz] between 1S0 and 3P1 mJ = 1, radial(3p1-1S0):") 
print((oscillator_freq_r(laserpower,beam_w0,pol3P1_mj1_Pi_785,atomic_mass)-oscillator_freq_r(laserpower,beam_w0,pol1S0_785,atomic_mass))/1000)
print("Trap frequency difference [kHz] between 1S0 and 3P1 mJ = 1, axial (3p1-1S0):") 
print((oscillator_freq_z(laserpower,beam_w0,785e-9,pol3P1_mj1_Pi_785,atomic_mass)-oscillator_freq_z(laserpower,beam_w0,785e-9,pol1S0_785,atomic_mass))/1000)





print("For 813 nm")
print("Trap depth of 1S0(microK,kHz)=")
print(potential_min(laserpower,beam_w0,pol1S0_813)*1e6)
print(potential_min(laserpower,beam_w0,pol1S0_813)*1e6*20.837)
print("Trap depth of 3P1, mj=0(microK,kHz)=")
print(potential_min(laserpower,beam_w0,pol3P1_mj0_Pi_813)*1e6)
print(potential_min(laserpower,beam_w0,pol1S0_813)*1e6*20.837)
print("Trap depth of 3P1, mj=1(microK,kHz)=")
print(potential_min(laserpower,beam_w0,pol3P1_mj1_Pi_813)*1e6)
print(potential_min(laserpower,beam_w0,pol3P1_mj1_Pi_813)*1e6*20.837)

print("difference at center of trap (AC stark shift) [kHz][3P1mj=0-1S0]=")
print((potential_min(laserpower,beam_w0,pol3P1_mj0_Pi_813)-potential_min(laserpower,beam_w0,pol1S0_813))*1e6*20.837)


print("difference at center of trap [kHz][3P1mj=1-1S0]=")
print((potential_min(laserpower,beam_w0,pol3P1_mj1_Pi_813)-potential_min(laserpower,beam_w0,pol1S0_813))*1e6*20.837)
# print("Trap frequency  [kHz] between 1S0, radial:") 
# print(oscillator_freq_r(laserpower,beam_w0,pol1S0_813,atomic_mass)/1000) 
# print("Trap frequency  [kHz] between 1S0, axial:") 
# print(oscillator_freq_z(laserpower,beam_w0,813e-9,pol1S0_813,atomic_mass)/1000)


# ### Let's check how far atoms move after different time of flight


# time_of_flight = np.linspace(0,500,100)*1e-6
# position_afterwards = -0.5*gEarth*np.power(time_of_flight,2)

# plt.plot(time_of_flight*1e6,position_afterwards*1e6)
# plt.xlabel("Time of flight [micro s]")
# plt.ylabel("Position [microns]")
# plt.show()

print("For 532 nm")
print("Trap depth of 1S0(microK,kHz)=")
print(potential_min(laserpower,beam_w0,pol1S0_532)*1e6)
print(potential_min(laserpower,beam_w0,pol1S0_532)*1e6*20.837)
print("Trap depth of 3P1, mj=0(microK,kHz)=")
print(potential_min(laserpower,beam_w0,pol3P1_mj0_Pi_532)*1e6)
print(potential_min(laserpower,beam_w0,pol3P1_mj0_Pi_532)*1e6*20.837)
print("Trap depth of 3P1, mj=1(microK,kHz)=")
print(potential_min(laserpower,beam_w0,pol3P1_mj1_Pi_532)*1e6)
print(potential_min(laserpower,beam_w0,pol3P1_mj1_Pi_532)*1e6*20.837)

print("difference at center of trap (AC starkshift) [kHz][3P1mj=0-1S0]=")
print((potential_min(laserpower,beam_w0,pol3P1_mj0_Pi_532)-potential_min(laserpower,beam_w0,pol1S0_532))*1e6*20.837)


print("difference at center of trap (AC starkshift) [kHz][3P1mj=1-1S0]=")
print((potential_min(laserpower,beam_w0,pol3P1_mj1_Pi_532)-potential_min(laserpower,beam_w0,pol1S0_532))*1e6*20.837)
import matplotlib.pyplot as plt 
import numpy as np

"""plots experimental sequence of the red MOT, as well as the relative amount of atoms

there are a lot of different timestamps, which i will summarize here:
	t_0: the 461 nm (blue) MOT beams are shut off and the gradient field
    is sharply jumped from 55 G/cm to 1.48 G/cm. 

	At this point, the red beams have been on for 90 ms, and the frequency is scanned 
    from -100 kHz to -1.72 MHz using a 45 kHz scan frequency. 
	After 23 ms, the cloud is compressed by linearly ramping up
    the gradient field to 4.24 G/cm. 

	$t_1$: subsequently, the modulation depth is reduced, to prepare for
    the transition to single frequency operation. 

	$t_2$, the scanning is switched to single-frequency operation. 

	Over a duration of 73 ms, The intensity and detuning are exponentially decreased until 
    they reach the values of the final red MOT stage at $t=t_4$. 
	The final stage lasts 30 ms.
"""

# %% time
# t parameters
t_blue = 20
t_cap= 23
t_ramp = 73
t_bb1 = t_cap + t_ramp
t_bb2 = 46
t_sf1 = 144
t_sf2 = 5

# derived parameters
t_bb = t_bb1 + t_bb2
t_sf = t_sf1 + t_sf2
time = np.linspace(-t_blue, t_bb + t_sf, 1000)

# %% intialize plot

fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows = 5, ncols = 1,
                                          figsize = (5, 9), sharex = True)
ax5.set_xlabel('time [ms]')

# %% blue MOT beams

blue_beams = np.piecewise(time, [time < 0, time >= 0], [1, 0])

ax1.plot(time, blue_beams, label = 'blue beams', color = 'blue')
ax1.grid()
ax1.sharex(ax2)
y_ticks = [0, 1]
y_tick_labels = ['off', 'on']
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels)
ax1.legend()

# %% quadropole field gradient

# B parameters
b_blue = 55
b_bb1 = 1.48
b_bb2 = 4.24
b_ramp = (b_bb2 - b_bb1)/t_ramp*(time - t_cap) + b_bb1


def grad_field(t):
    if t < 0:
        return b_blue
    elif (t>=0) & (t<t_cap):
        return b_bb1
    elif (t>=t_cap) & (t<t_bb1):
        return (b_bb2 - b_bb1)/t_ramp*(t - t_cap) + b_bb1
    else:
        return b_bb2


grad_field = [grad_field(t) for t in time]

ax2.grid()
ax2.plot(time, grad_field, color = 'black', label = 'magnetic field gradient')
ax2.set_ylabel('Gradient [G/cm]')
ax2.legend()

# %% red MOT beams intensity

# red intensity parameters
i_bb1 = 2
i_bb2 = 2
i_dec_start = 0.5
i_dec_end = 0.03
i_dec_tau = 20
i_dec_time = 144


def red_intensity(t):
    if t < t_bb1:
        return i_bb1
    elif (t >= t_bb1) & (t < t_bb1+t_bb2):
        return i_bb2
    elif (t >= t_bb1+t_bb2) & (t < t_bb1+t_bb2+i_dec_time):
        pre_factor = (i_dec_end - i_dec_start)/(np.exp(-t_sf1/i_dec_tau) - 1)
        value = pre_factor*(np.exp(-(t - t_bb1 - t_bb2)/i_dec_tau) - np.exp(-t_sf1/i_dec_tau)) + i_dec_end
        return value
    else:
        return i_dec_end
    

red_intensity = [red_intensity(t) for t in time]

ax3.grid()
ax3.plot(time, red_intensity, color = 'red', label = '689 intensity')
ax3.set_yscale('log')
ax3.set_ylabel('Intensity [mW]')
ax3.legend()

# %% red MOT beams frequency

# bb stages
detuning_bb1 = -189*1e3
moddepth_bb1 = 1.22*1e6
moddepth_bb2 = 0.788*1e6

# sf stages
freq_dec_start = -421*1e3
freq_dec_tau =20
freq_dec_end = -100e3


def red_frequency(t):
    if t < t_bb1:
        return -moddepth_bb1
    elif (t >= t_bb1) & (t < t_bb1+t_bb2):
        return -moddepth_bb2
    elif (t >= t_bb1+t_bb2) & (t < t_bb1+t_bb2+i_dec_time):
        pre_factor = (freq_dec_end - freq_dec_start)/(np.exp(-t_sf1/freq_dec_tau) - 1)
        value = pre_factor*(np.exp(-(t - t_bb1 - t_bb2)/freq_dec_tau) - np.exp(-t_sf1/freq_dec_tau)) + freq_dec_end
        return value
    else:
        return freq_dec_end
    

red_freqencies = np.array([red_frequency(t) for t in time])

ax4.grid()
ax4.plot(time, red_freqencies/1e6, color = 'red', label = '689 detuning')
ax4.hlines(y = detuning_bb1/1e6, xmin = -t_blue, xmax = t_bb1 + t_bb2, color = 'red')
ax4.vlines(x = t_bb1 + t_bb2, ymin = -moddepth_bb2/1e6, ymax = detuning_bb1/1e6, color = 'red')
ax4.set_ylabel('Detuning [MHz]')
ax4.legend()

# %% plot atom number

times_atom_nr = [0, t_cap, t_bb1, t_bb, t_bb + t_sf1, t_bb + t_sf]
atom_nr= [145934.30376075473, 53684.178798866975, 43024.98826138505,
          40680.60235048706, 25054.22079618794, 7458.480563721959]

# normalize
atom_nr = atom_nr/np.max(atom_nr)

# labels
labels = ['blue MOT', 'bMOT capture', 'bMOT compressed', 'bMOT decreased mod. depth',
          'sMOT decay', 'sMOT final']
"""
ax5.grid()
ax5.scatter(times_atom_nr, atom_nr)
ax5.set_ylabel('rel. atom number')
"""


# %% plotting

# vertical lines
for ax_i in (ax1, ax2, ax3, ax4, ax5):
    ax_i.axvline(0, color = 'grey', linestyle = '--')
    ax_i.axvline(t_bb1, color='grey', linestyle='--')
    ax_i.axvline(t_bb, color='grey', linestyle='--')
    ax_i.axvline(t_bb + t_sf1, color='grey', linestyle='--')

# labels for the various stages
# Add labels for the various stages
ax1.text(0, ax1.get_ylim()[1], r'$t_0$', color='black', ha='center', va='bottom')
ax1.text(t_bb1, ax1.get_ylim()[1], r'$t_1$', color='black', ha='center', va='bottom')
ax1.text(t_bb, ax1.get_ylim()[1], r'$t_2$', color='black', ha='center', va='bottom')
ax1.text(t_bb + t_sf1, ax1.get_ylim()[1], r'$t_3$', color='black', ha='center', va='bottom')

plt.show()
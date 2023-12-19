import matplotlib.pyplot as plt 
import numpy as np

# t parameters
t_blue = 20
t_cap= 23
t_ramp = 73
t_bb1 = t_cap + t_ramp
t_bb2 = 20
t_sf1 = 73
t_sf2 = 30

# derived parameters
t_bb = t_bb1 + t_bb2
t_sf = t_sf1 + t_sf2
time = np.linspace(-t_blue, t_bb + t_sf, 1000)

# B parameters
b_blue = 55
b_bb1 = 1.48
b_bb2 = 4.24
b_ramp = (b_bb2 - b_bb1)/t_ramp*(time - t_cap) + b_bb1

# red intensity parameters
i_bb1 = 0.4
i_bb2 = 0.4
i_dec_start = 0.0259
i_dec_end = 0.0033
i_dec_tau = 16.8
i_dec_time = 73

# entries plot

blue_beams = np.piecewise(time, [time < 0, time >= 0], [1, 0])


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


def red_intensity(t):
    if t < t_bb1:
        return i_bb1
    elif (t >= t_bb1) & (t < t_bb1+t_bb2):
        return i_bb2
    elif (t >= t_bb1+t_bb2) & (t < t_bb1+t_bb2+i_dec_time):
        return i_dec_start*np.exp(-(t-t_bb1-t_bb2)/i_dec_tau)
    else:
        return 0
    
red_intensity = [red_intensity(t) for t in time]


# plotting
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(time, blue_beams)
ax1.grid()
ax1.sharex(ax2)

ax2.grid()
ax2.plot(time, grad_field, color = 'black')

ax3.grid()
ax3.plot(time, red_intensity, color = 'red')
plt.show()
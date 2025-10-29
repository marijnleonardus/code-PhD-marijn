"first run analyze_uniformity.py to get the data"

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# add local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../modules', '../../utils'])

# user defined libraries
from fitting_functions_class import FittingFunctions
from plot_utils import Plotting
from units import MHz, kHz

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

#######################################
# UNIFORMITY
#######################################

# import data
loaded_data_uniformity = np.load('output/combined_figs/uniformity_data_dict.npz')

# --- Loading ---
sem_surv_prob_uni = loaded_data_uniformity['sem_surv_prob']
x_axis_fit_uni = loaded_data_uniformity['x_axis_fit']
surv_prob_uni = loaded_data_uniformity['surv_prob']
x_grid_uni = loaded_data_uniformity['x_grid']  
popt_list_uni = loaded_data_uniformity['popt_list']
detunings = loaded_data_uniformity['detunings']

roi_idx_chosen = int(22)
popt_uni = popt_list_uni[roi_idx_chosen]

detunings_2d = detunings.reshape(5, 5)/MHz

#######################################
# TRREQUENCY, AXIAL
#######################################
loaded_data_trap_freq_ax = np.load('output/combined_figs/trap_freq_data_dict_axial.npz')

x_grid_trapfreq_ax = loaded_data_trap_freq_ax['x_grid']
glob_surv_trapfreq_ax = loaded_data_trap_freq_ax['glob_surv']
glob_surv_sem_trapfreq_ax = loaded_data_trap_freq_ax['glob_surv_sem'] 
x_axis_fit_trapfreq_ax = loaded_data_trap_freq_ax['x_axis_fit']
popt_trapfreq_ax = loaded_data_trap_freq_ax['popt']

#######################################
# TRREQUENCY, RADIAL
#######################################
loaded_data_trap_freq_rad = np.load('output/combined_figs/trap_freq_data_dict_rad.npz')

x_grid_trapfreq_rad = loaded_data_trap_freq_rad['x_grid']
glob_surv_trapfreq_rad = loaded_data_trap_freq_rad['glob_surv']
glob_surv_sem_trapfreq_rad = loaded_data_trap_freq_rad['glob_surv_sem'] 
x_axis_fit_trapfreq_rad = loaded_data_trap_freq_rad['x_axis_fit']
popt_trapfreq_rad = loaded_data_trap_freq_rad['popt']

#######################################
# plotting
########################################

# print one of the fits to show how you did the data analysis
fig_width = 5  # inches, matches two columns
fig_height = fig_width*0.8

fig = plt.figure(figsize=(fig_width, fig_height))
gs = GridSpec(2, 2, figure=fig, wspace=0.2, hspace=0.5)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1], sharey=ax3)

roi_idx_pick = int(22)
ax1.errorbar(x_grid_uni/MHz, surv_prob_uni[roi_idx_chosen, :], sem_surv_prob_uni[roi_idx_chosen, :], fmt='o', color='blue')
ax1.plot(x_axis_fit_uni/MHz, FittingFunctions.gaussian_function(x_axis_fit_uni, *popt_uni), color='red')
ax1.set_xlabel('Detuning [MHz]')
ax1.set_ylabel('Survival probabiility')

# plot heatmap
im = ax2.imshow(
    detunings_2d, 
    origin='lower',      # y=0 at bottom
    cmap='cool',         # similar to your magenta–cyan colormap
    interpolation='none' # no smoothing, keep blocky grid
)
cbar = fig.colorbar(im, ax=ax2)
cbar.set_label(r' AC Stark Shift [MHz]')
ax2.set_xlabel('x index')
ax2.set_ylabel('y index')

# Add grid lines
ax2.set_xticks(np.arange(detunings_2d.shape[1] + 1) - 0.5, minor=True)
ax2.set_yticks(np.arange(detunings_2d.shape[0] + 1) - 0.5, minor=True)
ax2.grid(which="minor", color="w", linestyle='-', linewidth=1)
ax2.tick_params(which="minor", bottom=False, left=False)  # hide minor tick marks
ax2.tick_params(which="major", bottom=False, left=False)  # hide major tick marks

# plot trap freq (ax and rad)
ax3.errorbar(x_grid_trapfreq_ax/kHz, glob_surv_trapfreq_ax, yerr=glob_surv_sem_trapfreq_ax, fmt='o', color='blue')
ax3.plot(x_axis_fit_trapfreq_ax/kHz, FittingFunctions.lorentzian(x_axis_fit_trapfreq_ax, *popt_trapfreq_ax), color='red')

ax4.errorbar(x_grid_trapfreq_rad/kHz, glob_surv_trapfreq_rad, yerr=glob_surv_sem_trapfreq_rad, fmt='o', color='blue')
ax4.plot(x_axis_fit_trapfreq_rad/kHz, FittingFunctions.lorentzian(x_axis_fit_trapfreq_rad, *popt_trapfreq_rad), color='red')

# cosmetics to get the broken axis look
ax3.spines['right'].set_visible(False)
ax4.spines['left'].set_visible(False)

ax4.tick_params(labelleft=False)

# Remove tick labels between plots
ax4.yaxis.tick_right()
ax3.tick_params(labelright=False)
ax4.tick_params(labelleft=False)

# Add diagonal lines to indicate the break
d = 0.015  # size of diagonal break lines
kwargs = dict(transform=ax3.transAxes, color='k', clip_on=False)
ax3.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=ax4.transAxes)  # switch to the other axes
ax4.plot((-d, +d), (-d, +d), **kwargs)
ax4.plot((-d, +d), (1 - d, 1 + d), **kwargs)

fig.text(0.5, 0.02, 'Modulation freq. [kHz]', ha='center')
ax3.set_ylabel('Average survival probability')

# subplots annotations
ax1.text(-0.35, 0.85, r'\textbf{a)}', transform=ax1.transAxes, size=8.5, usetex=True)
ax1.text(-0.30, 0.85, r'\textbf{b)}', transform=ax2.transAxes, size=8.5, usetex=True)
ax1.text(-0.35, 0.85, r'\textbf{c)}', transform=ax3.transAxes, size=8.5, usetex=True)

# save plots
Plot = Plotting('output')
Plot.savefig('tweezers_characterization_combined.pdf')
plt.show()

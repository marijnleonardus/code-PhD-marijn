# Marijn Venderbosch
# dec 2022 - oct 2023

"""simulates trajectories of Sr88 atoms in red MOT

adapted from PyLCP example: 
https://python-laser-cooling-physics.readthedocs.io/en/latest/examples/MOTs/10_recoil_limited_MOT.html

Has been tested in a pyton 3.7 anaconda environment on a Linux machine
The following packages were installed
- pylcp
- matplotlib
- pathos
- scipy
"""

# %% imports 

# standard libraries
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylcp 
import scipy.constants 
from pylcp.common import progressBar
from scipy.constants import hbar, pi, Boltzmann, proton_mass
import pathos 
from matplotlib.patches import Ellipse

# user defined libraries
from modules.plotting import Plotting
from modules.magnetic_field_class import QuadrupoleYMagneticField
from modules.laser_beam_class import AngledMOTBeams

# %% constants

# parameters
b_gauss = 4.24  # Gauss
saturation = 80 
detuning = -120e3  # Hz
simulation_time = 0.1  # s
nr_atoms = 2500
nr_nodes = 4  # nr cores for multithreading

# constants
wavelength = 689e-9  # m
linewidth = 7.4e3  # rad/s
bohr_magneton = scipy.constants.value('Bohr magneton in Hz/T')
print(bohr_magneton)
atomic_mass_unit = scipy.constants.value('atomic mass constant')

# derived constants
k = 2*pi/wavelength  # m^-1
x0 = 1/k  # our length scale in m
gamma = 2*pi*linewidth  # Hz
t0 = 1/gamma  # our time scale in s
mass = 88*atomic_mass_unit*x0**2/hbar/t0  # unitless mass
g = np.array([0., 0., -9.8*t0**2/x0])  # unitless gravity
tmax = simulation_time/t0  # dimensionless time
det = detuning/linewidth  # dimensionless detuning

# derived parameter
# Magnetic field gradient parameter (factor of 3/2 from excited state g-factor.)
b_tesla = b_gauss*1e-4
alpha = (3/2)*bohr_magneton*b_tesla*(x0*1e2)/linewidth


# %% define magnetic field, laser beams, governing equation

# quadrupole field
magField = QuadrupoleYMagneticField(alpha)

# laser beams
# start with 4 diagonal beams
laserBeams = AngledMOTBeams(delta = det, s = saturation,  beam_type = pylcp.infinitePlaneWaveBeam)

# add 2 horizontal beams
HorizontalBeam1 = pylcp.laserBeam(kvec=np.array([0, +1., 0.]), delta = det, pol=+1, s=0.5*saturation)
HorizontalBeam2 = pylcp.laserBeam(kvec=np.array([0.,-1., 0.]), delta = det, pol=+1, s=0.5*saturation)
laserBeams.add_laser(HorizontalBeam1)
laserBeams.add_laser(HorizontalBeam2)

# define hamiltonian
Hg, mugq = pylcp.hamiltonians.singleF(F=0, muB=1)
He, mueq = pylcp.hamiltonians.singleF(F=1, muB=1)
dq = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)
hamiltonian = pylcp.hamiltonian(Hg, He, mugq, mueq, dq, mass=mass)

# define equation
eqn = pylcp.rateeq(laserBeams, magField, hamiltonian, g)
# uncomment for the heuristic equation
#eqn = pylcp.heuristiceq(laserBeams, magField, g, mass=mass)

# %% plot force profiles

Plot = Plotting(length_scale=x0, time_scale=t0)
Plot.force_profile_3d(eqn)

# %% dynamics

Plot.trajectory_single_atom(eqn, tmax)

# %% simulate many atoms


def generate_random_solution(args):
    """for each atom index generate a unique seed, then generate
    an array of random numbers and evolve the solution"""

    atom_index, x = args

    # We need to generate random numbers to prevent solutions from being seeded
    # with the same random number.
    #np.random.seed(atom_index) 

    # generate random nr between -100 and 100 um to set as intial position (x,y,z)
    initial_position = np.random.uniform(-100/(x0*1e6), 100/(x0*1e6), size=3)  
    eqn.set_initial_position(initial_position)

    # generate random start velocity (v0x, v0y, v0z) in units of Gamma/k
    initial_velocity = np.random.uniform(-5, 5, 3)
    eqn.set_initial_velocity(initial_velocity)

    eqn.evolve_motion([0, tmax], t_eval=np.linspace(0, tmax, 10001),
        random_recoil=True, progress_bar=False, max_step=1.)
    return eqn.sol


def run_parallel(nr_atoms, nr_nodes):
    """Uses parallel processing to generate random solutions

    it creates a ProcessPool with 4 nodes and splits the task of 
    generating random solutions into chunks that are processed in parallel
    
    returns:
    - a list of solutions
    """
    
    chunksize = 4
    sols = []
    progress = progressBar()

    with pathos.pools.ProcessPool(nodes=nr_nodes) as pool:
        args_list = [(atom_index, chunksize) for atom_index in range(0, nr_atoms, chunksize)]
        sols = pool.map(generate_random_solution, args_list)
        progress.update(1.0)
    return sols


def ejection_criterion(solution_list):
    """ ejection criterion, if the position is larger than 500,
    the atom is said to be ejected"""

    ejected = [np.bitwise_or(
        np.abs(solution.r[0, -1]*(1e6*x0)) > 500,
        np.abs(solution.r[1, -1]*(1e6*x0)) > 500
    ) for solution in solution_list]
    return ejected


sols = run_parallel(nr_atoms=nr_atoms, nr_nodes=nr_nodes)
ejected = ejection_criterion(sols)

# %% plot trajectories and calculate temperatures

# empty matrices to later fill with trajectories of non-ejected atoms
# for the temperature calculations
allx = allz = np.array([], dtype='float64')
allvx = allvz = np.array([], dtype='float64')

fig1, ax1 = plt.subplots(3, 2, figsize=(7, 2*2.75))
for sol, ejected_i in zip(sols, ejected):
    for ii in range(3):
        # if ejected, plot red, if not ejected plot blue
        if ejected_i:
            # plot ejected atoms in red
            ax1[ii, 0].plot(sol.t/1e3, sol.v[ii], color='r', linewidth=0.25)
            ax1[ii, 1].plot(sol.t/1e3, sol.r[ii]*alpha, color='r', linewidth=0.25)
        else:
            # plot non-ejected atoms in blue 
            ax1[ii, 0].plot(sol.t/1e3, sol.v[ii], color='b', linewidth=0.25)
            ax1[ii, 1].plot(sol.t/1e3, sol.r[ii]*alpha, color='b', linewidth=0.25) 
            
            # and log the velocities (temperature)
            allx = np.append(allx, sol.r[0][::1]*(1e6*x0)) # um
            allz = np.append(allz, sol.r[2][::1]*(1e6*x0)) # um
            allvx = np.append(allvx, sol.v[0][::10]) # k/gamma
            allvz = np.append(allvz, sol.v[2][::10]) # k/gamma

# velocity (left) plots settings
for ax_i in ax1[:, 0]:
    ax_i.set_ylim((-20, 20))
for ax_i in ax1[:, 1]:
    ax_i.set_ylim((-4., 4.))
for ax_i in ax1[-1, :]:
    ax_i.set_xlabel('$10^3 \Gamma t$')

# position (right) plots settings
for jj in range(2):
    for ax_i in ax1[jj, :]:
        ax_i.set_xticklabels('')
for ax_i, lbl in zip(ax1[:, 0], ['x','y','z']):
    ax_i.set_ylabel('$v_' + lbl + '/(\Gamma/k)$')
for ax_i, lbl in zip(ax1[:, 1], ['x','y','z']):
    ax_i.set_ylim((-80, 80))
    ax_i.set_ylabel('$\\alpha ' + lbl + '$')

fig1.subplots_adjust(left=0.1, bottom=0.08, wspace=0.4)


def calculate_temperature(array_gammaoverk):
    """calculate temperature from velocity distribution
    
    inputs:
    - array_gammaoverk (np.array of floats): velocity in units of gamma/k"""

    mass_sr = 88*proton_mass

    # convert to m/s
    velocity_ms = array_gammaoverk/(k/gamma)

    # calculate RMS value
    rms_velocity_ms = np.sqrt(np.mean(velocity_ms**2))

    # calculate temperature 
    temperature = mass_sr*rms_velocity_ms**2/Boltzmann
    temperature_uK = temperature*1e6
    print(temperature_uK)


calculate_temperature(allvx)
calculate_temperature(allvz)

# %% compute the 2d histogram 

# and normalize
img, x_edges, z_edges = np.histogram2d(allx, allz, 
    bins=[np.arange(-400, 400, 5.), np.arange(-400., 50., 5.)])
img = img/img.max()

fig2, ax2 = plt.subplots(figsize = (4, 3))
fig2.subplots_adjust(left=0.08, bottom=0.2, top=0.97, right=0.95)
ax2.set_ylabel('$z$ ($\mu$m)')
ax2.set_xlabel('$x$ ($\mu$m)')

# plot 
im = ax2.imshow(img.T, origin='lower', cmap='Reds', aspect='equal',
    extent=(np.amin(x_edges), np.amax(x_edges),
            np.amin(z_edges), np.amax(z_edges)))

# add ellipse
# height/width interchanged
width_semi_ax = 4*det/alpha*(1e6*x0)
height_semi_ax = 4*det/alpha*(1e6*x0)
ellip = Ellipse(xy = (0, 0), width = height_semi_ax, height = width_semi_ax, linestyle='--',
    linewidth=1, facecolor='none', edgecolor='blue', label='Zeeman shift equals detuning')
ax2.add_patch(ellip)
ax2.legend()

# colorbar
pos = ax2.get_position()
cbar_ax = fig2.add_axes([1, pos.y0, 0.015, pos.y1-pos.y0])
colorbar_ticks = [0, 0.5, 1.]
cbar = fig2.colorbar(im, cax=cbar_ax, ticks = colorbar_ticks)
cbar_ax.set_ylabel('Density (arb. units)')

plt.show()


# %%

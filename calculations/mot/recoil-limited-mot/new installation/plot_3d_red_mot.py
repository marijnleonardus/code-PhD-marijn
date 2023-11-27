# Marijn Venderbosch
# dec 2022 - oct 2023

"""simulates trajectories of Sr88 atoms in red MOT

adapted from PyLCP example: 
https://python-laser-cooling-physics.readthedocs.io/en/latest/examples/MOTs/10_recoil_limited_MOT.html
"""

# %% imports 

import numpy as np
import matplotlib.pyplot as plt
import pylcp 
import scipy.constants 
from pylcp.common import progressBar
from scipy.constants import hbar, pi
import pathos 

from matplotlib.patches import Ellipse

# %% constants

# parameters
b_gauss = 4.24  # Gauss
saturation = 25 
detuning = -200e3  # Hz
simulation_time = 0.5  # s

# constants
wavelength = 689e-9  # m
linewidth = 7.4e3  # rad/s
bohr_magneton = scipy.constants.value('Bohr magneton in Hz/T')
atomic_mass_unit = scipy.constants.value('atomic mass constant')

# derived constants
k = 2*pi/wavelength  # m^-1
x0 = 1/k  # our length scale in m
gamma = 2*pi*linewidth  # Hz
t0 = 1/gamma  # our time scale in s
mass = 88*atomic_mass_unit*x0**2/hbar/t0  # unitless mass
g = np.array([-9.8*t0**2/x0, 0., 0.])  # unitless gravity
tmax = simulation_time/t0  # dimensionless time
det = detuning/linewidth  # dimensionless detuning

# derived parameter
# Magnetic field gradient parameter (factor of 3/2 from excited state g-factor.)
b_tesla = b_gauss*1e-4
alpha = (3/2)*bohr_magneton*b_tesla*(x0*1e2)/linewidth

# %% setting up the parameters, magnetic field, lasers

magField = pylcp.quadrupoleMagneticField(alpha)
laserBeams = pylcp.conventional3DMOTBeams(delta=det, s=saturation, beam_type=pylcp.infinitePlaneWaveBeam)

Hg, mugq = pylcp.hamiltonians.singleF(F=0, muB=1)
He, mueq = pylcp.hamiltonians.singleF(F=1, muB=1)

dq = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)

hamiltonian = pylcp.hamiltonian(Hg, He, mugq, mueq, dq, mass=mass)

# uncomment for the heuristic equation
#eqn = pylcp.heuristiceq(laserBeams, magField, g, mass=mass)
eqn = pylcp.rateeq(laserBeams, magField, hamiltonian, g)

# %% force profile

# for plotting rescaling
length_mm = 1e3*x0  # mm
z = np.linspace(-0.4, 0.4, 101)/length_mm
r = np.array([np.zeros(z.shape), np.zeros(z.shape), z])
v = np.zeros((3,) + z.shape)

eqn.generate_force_profile(r, v, name='Fz')

# plot force profile
fig, ax = plt.subplots(1, 1)
ax.plot(z*length_mm, eqn.profile['Fz'].F[2])
ax.set_xlabel('$z$ (mm)')
ax.set_ylabel('$f/(\hbar k \Gamma)$')
plt.show()

# %% dynamics

if isinstance(eqn, pylcp.rateeq):
    eqn.set_initial_pop(np.array([1., 0., 0., 0.]))
eqn.set_initial_position(np.array([0., 0., 0.]))

# simulate for 10% of the time
eqn.evolve_motion([0, tmax*0.1], random_recoil=True, progress_bar=True, max_step=1.)

# plot test solution
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.75))
ax[0].plot(eqn.sol.t*t0, eqn.sol.r.T*(1e6*x0))
ax[1].plot(eqn.sol.t*t0, eqn.sol.v.T)
ax[0].set_ylabel('$r$ ($\mu$m)')
ax[0].set_xlabel('$t$ (s)')
ax[1].set_ylabel('$v/(\Gamma/k)$')
ax[1].set_xlabel('$t$ (s)')
fig.subplots_adjust(left=0.08, wspace=0.22)

# %% simulate many atoms


if hasattr(eqn, 'sol'):
    del eqn.sol


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
    generating random solutions into chunks
    
    Each chunk is processed in parallel using the map function
    
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


sols = run_parallel(nr_atoms=50, nr_nodes=8)

# ejection criterion, if the position is larger than 500, the atom is said to be ejected
ejected = [np.bitwise_or(
    np.abs(sol.r[0, -1]*(1e6*x0))>500,
    np.abs(sol.r[1, -1]*(1e6*x0))>500
) for sol in sols]

# %% plot trajectories

fig1, ax1 = plt.subplots(3, 2, figsize=(7, 2*2.75))
for sol, ejected_i in zip(sols, ejected):
    for ii in range(3):
        # if ejected, plot red, if not ejected plot blue
        if ejected_i:
            ax1[ii, 0].plot(sol.t/1e3, sol.v[ii], color='r', linewidth=0.25)
            ax1[ii, 1].plot(sol.t/1e3, sol.r[ii]*alpha, color='r', linewidth=0.25)
        else:
            ax1[ii, 0].plot(sol.t/1e3, sol.v[ii], color='b', linewidth=0.25)
            ax1[ii, 1].plot(sol.t/1e3, sol.r[ii]*alpha, color='b', linewidth=0.25)

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

# %% plot histogram 2d

# log coordinates at fixed points during trajectories
allx = allz = np.array([], dtype='float64')
for sol in sols:
    allx = np.append(allx, sol.r[0][::1]*(1e6*x0))
    allz = np.append(allz, sol.r[2][::1]*(1e6*x0))

# compute the 2d histogram and normalize
# switched x and z directions, because gravity is in the x direction, but
# should appear in z direction in plot
img, x_edges, z_edges = np.histogram2d(allz, allx, 
    bins=[np.arange(-300, 300, 5.), np.arange(-400., 0., 5.)])
img = img/img.max()

fig2, ax2 = plt.subplots(figsize = (4, 3))
fig2.subplots_adjust(left=0.08, bottom=0.12, top=0.97, right=0.9)
ax2.set_ylabel('$z$ ($\mu$m)')
ax2.set_xlabel('$x$ ($\mu$m)')

# plot 
im = ax2.imshow(img.T, origin='lower', cmap='Reds', aspect='equal',
    extent=(np.amin(x_edges), np.amax(x_edges),
            np.amin(z_edges), np.amax(z_edges)))

# add ellipse
# height/width interchanged
width_semi_ax = 4*det/alpha*(1e6*x0)
height_semi_ax = 2*det/alpha*(1e6*x0)
ellip = Ellipse(xy = (0, 0), width = height_semi_ax, height = width_semi_ax, linestyle='--',
    linewidth=1, facecolor='none', edgecolor='blue', label='Zeeman shift equals detuning')
ax2.add_patch(ellip)
ax2.legend()

# colorbar
pos = ax2.get_position()
cbar_ax = fig2.add_axes([0.91, pos.y0, 0.015, pos.y1-pos.y0])
cbar = fig.colorbar(im, cax=cbar_ax, ticks = [0., 0.5, 1.])
cbar_ax.set_ylabel('Density (arb. units)')

plt.show()

# %%

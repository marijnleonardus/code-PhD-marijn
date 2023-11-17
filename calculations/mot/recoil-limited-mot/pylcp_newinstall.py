# Marijn Venderbosch
# dec 2022 - oct 2023

"""simulates trajectories of Sr88 atoms in red MOT

adapted from PyLCP example: 
https://python-laser-cooling-physics.readthedocs.io/en/latest/examples/MOTs/10_recoil_limited_MOT.html
"""
import numpy as np
import matplotlib.pyplot as plt
import pylcp
import scipy.constants as cts
from pylcp.common import progressBar
from scipy.constants import hbar, pi
import pathos 

# %% constants

k = 2*pi/689e-7    # cm^{-1}
x0 = 1/k              # our length scale in cm
gamma = 2*pi*7.5e3 # 7.5 kHz linewidth
t0 = 1/gamma          # our time scale in s

# Magnetic field gradient parameter (the factor of 3/2 comes from the
# excited state g-factor.)
alpha = (3/2)*cts.value('Bohr magneton in Hz/T')*1e-4*8*x0/7.5E3

# The unitless mass parameter:
mass = 87.8*cts.value('atomic mass constant')*(x0*1e-2)**2/hbar/t0

# Gravity
g=9.81
g = -np.array([0., 0., g*t0**2/(x0*1e-2)])

print(x0, t0, mass, alpha, g)

# %%

sat = 25
det = -200/7.5

magField = pylcp.quadrupoleMagneticField(alpha)

laserBeams = pylcp.conventional3DMOTBeams(delta=det, s=sat, beam_type=pylcp.infinitePlaneWaveBeam)

Hg, mugq = pylcp.hamiltonians.singleF(F=0, muB=1)
He, mueq = pylcp.hamiltonians.singleF(F=1, muB=1)

dq = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)

hamiltonian = pylcp.hamiltonian(Hg, He, mugq, mueq, dq, mass=mass)

#eqn = pylcp.heuristiceq(laserBeams, magField, g, mass=mass)
eqn = pylcp.rateeq(laserBeams, magField, hamiltonian, g)

# %% force profile

z = np.linspace(-0.2, 0.2, 101)/(10*x0)

R = np.array([np.zeros(z.shape), np.zeros(z.shape), z])
V = np.zeros((3,) + z.shape)

eqn.generate_force_profile(R, V, name='Fz')

fig, ax = plt.subplots(1, 1)

ax.plot(z*(10*x0), eqn.profile['Fz'].F[2])
ax.set_xlabel('$z$ (mm)')
ax.set_ylabel('$f/(\hbar k \Gamma)$')


# %% dynamics

tmax = 0.4/t0
if isinstance(eqn, pylcp.rateeq):
    eqn.set_initial_pop(np.array([1., 0., 0., 0.]))
eqn.set_initial_position(np.array([0., 0., 0.]))
eqn.evolve_motion([0, tmax], random_recoil=True, progress_bar=True, max_step=1.)

# plot test solution
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.75))
ax[0].plot(eqn.sol.t*t0, eqn.sol.r.T*(1e4*x0))
ax[1].plot(eqn.sol.t*t0, eqn.sol.v.T)
ax[0].set_ylabel('$r$ ($\mu$m)')
ax[0].set_xlabel('$t$ (s)')
ax[1].set_ylabel('$v/(\Gamma/k)$')
ax[1].set_xlabel('$t$ (s)')
fig.subplots_adjust(left=0.08, wspace=0.22)

# %% simulate many atoms

if hasattr(eqn, 'sol'):
    del eqn.sol

def generate_random_solution(x, eqn=eqn, tmax=tmax):
    # We need to generate random numbers to prevent solutions from being seeded
    # with the same random number.
    np.random.seed(atom_index)  # Set a unique seed for each atom

    np.random.rand(256*x)
    eqn.evolve_motion(
        [0, tmax],
        t_eval=np.linspace(0, tmax, 1001),
        random_recoil=True,
        progress_bar=False,
        max_step=1.
    )

    return eqn.sol


def run_parallel():

    Natoms = 30
    chunksize = 2
    sols = []
    progress = progressBar()

    for jj in range(int(Natoms/chunksize)):
        with pathos.pools.ProcessPool(nodes=4) as pool:
            sols += pool.map(generate_random_solution, range(chunksize))
        progress.update((jj+1)/(Natoms/chunksize))
    
    # ejection criterion
    ejected = [np.bitwise_or(
        np.abs(sol.r[0, -1]*(1e4*x0))>500,
        np.abs(sol.r[1, -1]*(1e4*x0))>500
    ) for sol in sols]

    fig, ax = plt.subplots(3, 2, figsize=(6.25, 2*2.75))
    for sol, ejected_i in zip(sols, ejected):
        for ii in range(3):
            if ejected_i:
                ax[ii, 0].plot(sol.t/1e3, sol.v[ii], color='r', linewidth=0.25)
                ax[ii, 1].plot(sol.t/1e3, sol.r[ii]*alpha, color='r', linewidth=0.25)
            else:
                ax[ii, 0].plot(sol.t/1e3, sol.v[ii], color='b', linewidth=0.25)
                ax[ii, 1].plot(sol.t/1e3, sol.r[ii]*alpha, color='b', linewidth=0.25)

    for ax_i in ax[:, 0]:
        ax_i.set_ylim((-0.75, 0.75))
    for ax_i in ax[:, 1]:
        ax_i.set_ylim((-4., 4.))
    for ax_i in ax[-1, :]:
        ax_i.set_xlabel('$10^3 \Gamma t$')
    for jj in range(2):
        for ax_i in ax[jj, :]:
            ax_i.set_xticklabels('')
    for ax_i, lbl in zip(ax[:, 0], ['x','y','z']):
        ax_i.set_ylabel('$v_' + lbl + '/(\Gamma/k)$')
    for ax_i, lbl in zip(ax[:, 1], ['x','y','z']):
        ax_i.set_ylim((-100, 100))
        ax_i.set_ylabel('$\\alpha ' + lbl + '$')
    fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.22)

    allx = np.array([], dtype='float64')
    allz = np.array([], dtype='float64')

    for sol in sols:
        allx = np.append(allx, sol.r[0][::10]*(1e4*x0))
        allz = np.append(allz, sol.r[2][::10]*(1e4*x0))

    img, x_edges, z_edges = np.histogram2d(allx, allz, bins=[np.arange(-375, 376, 5.), np.arange(-600., 11., 5.)])

    fig4, ax4 = plt.subplots(1, 1)
    im = ax4.imshow(img.T, origin='bottom',
                extent=(np.amin(x_edges), np.amax(x_edges),
                        np.amin(z_edges), np.amax(z_edges)),
                cmap='Blues',
                aspect='equal')
    plt.show()


if __name__ == '__main__':
    run_parallel()



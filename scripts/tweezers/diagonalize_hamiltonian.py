import numpy as np

# Constants (set these to your specific values)
hbar = 1.054571817e-34  # Planck's constant over 2pi in J*s
mu_B = 9.2740100783e-24 # Bohr magneton in J/T
g_J = 1.5             # Example value for the excited state g-factor (replace as needed)

# AC Stark shifts (replace with your values)
Delta_g = 0.0         # Ground state shift
Delta_e = 1.0         # Excited state scalar shift
Delta_t = 0.1         # Tensor shift

# Magnetic field components in Tesla (replace with your values)
Bx, By, Bz = 0.01, 0.0, 0.0

# Build the Hamiltonian matrix.
# Order: |g>, |e,-1>, |e,0>, |e,+1>
H = np.zeros((4, 4), dtype=complex)

# Ground state only gets AC Stark shift
H[0, 0] = Delta_g

# Excited state diagonal elements: include both AC Stark shifts and Zeeman shifts along z.
H[1, 1] = Delta_e + Delta_t - mu_B * g_J * hbar * Bz  # m = -1
H[2, 2] = Delta_e - 2 * Delta_t                        # m = 0 (no Zeeman shift along z)
H[3, 3] = Delta_e + Delta_t + mu_B * g_J * hbar * Bz     # m = +1

# Off-diagonal Zeeman couplings from Jx and Jy.
# The matrix elements (in the |e,m> basis) are:
# <e,-1|H_B|e,0> = mu_B*g_J*(hbar/√2)*(B_x - i B_y)
# <e,0|H_B|e,+1> = mu_B*g_J*(hbar/√2)*(B_x - i B_y)
# And their Hermitian conjugates.
coupling = mu_B * g_J * hbar / np.sqrt(2)
H[1, 2] = coupling * (Bx - 1j * By)
H[2, 1] = np.conjugate(H[1, 2])
H[2, 3] = coupling * (Bx - 1j * By)
H[3, 2] = np.conjugate(H[2, 3])

# Diagonalize the Hamiltonian
eigenvalues, eigenvectors = np.linalg.eigh(H)

# Output the eigenvalues and eigenvectors
print("Eigenvalues (in energy units):")
print(eigenvalues)


from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.gate import X, Z

# Define the initial qubit state |1>
qubit_circuit1 = Qubit('1')

# Apply the X gate (Pauli-X) to the qubit, then the Z gate
x_applied = qapply(X(0) * qubit_circuit1)  # Apply X gate
zx_applied = qapply(Z(0) * x_applied)      # Apply Z gate to the result
print(zx_applied)

# Define another qubit state |1> for the second circuit
qubit_circuit2 = Qubit('1')

# Apply the Z gate (Pauli-Z) to the qubit, then the X gate
z_applied = qapply(Z(0) * qubit_circuit2)  # Apply Z gate
xz_applied = qapply(X(0) * z_applied)      # Apply X gate to the result
print(xz_applied)
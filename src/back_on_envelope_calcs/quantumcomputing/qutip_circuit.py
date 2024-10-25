# author Marijn Venderbosch
# March 2023

"""constructs CNOT gate from CZ gate and Hadamard gates"""

from qutip.qip.circuit import QubitCircuit
from qutip.qip.operations import gate_sequence_product

# initialize circuit
qc = QubitCircuit(N=2, num_cbits=0)

# construct circuit
qc.add_gate("SNOT", targets=1)
qc.add_gate("CZ", controls=0, targets=1)
qc.add_gate("SNOT", targets=1)

# print matrix of circuit
u_list = qc.propagators()
print(gate_sequence_product(u_list))

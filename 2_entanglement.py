#!/usr/bin/python3

from matplotlib import pyplot as plt
import qiskit
from qiskit_aer.primitives import SamplerV2
from qiskit.visualization import plot_histogram

# Create a register of 2 qubits
myQRegister = qiskit.QuantumRegister(2, '\psi')

# Create a register of 2 classical bits
myCRegister = qiskit.ClassicalRegister(2,'ClassicalBits')

# Create a quantum circuit with using myRegister
myCircuit = qiskit.QuantumCircuit(myQRegister, myCRegister)

# Hadamard gates on first qubit
myCircuit.h(0)
# CNOT gate controlled by first qubit on second qubit
myCircuit.cx(0,1)

# Measure all the qubits in myQRegister and store state in myCRegister
myCircuit.measure(myQRegister,myCRegister)

# Simulate the circuit
sampler = SamplerV2()
job = sampler.run([myCircuit],shots=2**15)
result = job.result()[0].data.ClassicalBits.get_counts()

# Plot a bar chart of all the results
plot_histogram(result, title='Bell State')
plt.show()

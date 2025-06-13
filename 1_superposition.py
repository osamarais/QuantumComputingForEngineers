#!/usr/bin/python3

from matplotlib import pyplot as plt
import qiskit
from qiskit_aer.primitives import SamplerV2
from qiskit.visualization import plot_histogram

# Create a register of 3 qubits
myQRegister = qiskit.QuantumRegister(3, '\psi')

# Create a register of 3 classical bits
myCRegister = qiskit.ClassicalRegister(3,'ClassicalBits')

# Create a quantum circuit with using myRegister
myCircuit = qiskit.QuantumCircuit(myQRegister, myCRegister)

# Hadamard gates on al qubits
myCircuit.h(myQRegister)

# Measure all the qubits in myQRegister and store state in myCRegister
myCircuit.measure(myQRegister,myCRegister)


# Simulate the circuit
sampler = SamplerV2()
job = sampler.run([myCircuit],shots=2**15)
result = job.result()[0].data.ClassicalBits.get_counts()

# Plot a bar chart of all the results
plot_histogram(result, title='Uniform Superposition')
plt.show()
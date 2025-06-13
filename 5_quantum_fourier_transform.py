#!/usr/bin/python3

from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit_aer.primitives import SamplerV2
from qiskit.visualization import plot_histogram

beforeFT = QuantumCircuit(5)
# Initialize state in uniform superposition
beforeFT.h([0,1,2,3,4])

# Measure all qubits
beforeFT.measure_all()

afterFT = QuantumCircuit(5)
# Initialize qubits
afterFT.h([0,1,2,3,4])

# Add Fourier transform operation
qft = QFT(num_qubits=5,do_swaps=False).to_gate()
afterFT.append(qft, qargs=[0,1,2,3,4])

# Measure all qubits
afterFT.measure_all()

# Decompose Fourier transform operation into gates for simulator
afterFT = afterFT.decompose(reps=2)

# Simulate the circuit

# Simulate the circuit
sampler = SamplerV2()
job = sampler.run([beforeFT,afterFT],shots=2**20)
result_before = job.result()[0].data.meas.get_counts()
result_after = job.result()[1].data.meas.get_counts()

# Plot a bar chart of all the results
plot_histogram(result_before,bar_labels=False,title='Before QFT')
plot_histogram(result_after,bar_labels=False,title='After QFT')

plt.show()

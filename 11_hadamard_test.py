import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import SGate
from qiskit.circuit import Parameter

# Since we will be executing this circuit for various alpha values
# define alpha as a parameter
alpha = Parameter('alpha')
sweeps = 20
_alpha = np.linspace(0,np.pi*2, sweeps)
params = np.vstack([_alpha]).T

# Number of shots to estimate values
shots1 = 1000
shots2 = 100_000

# Set up classical and quantum registers
qregister = QuantumRegister(2)
cregister = ClassicalRegister(1,'classical')

# Create Hadamard test circuit for real part
re_circuit = QuantumCircuit(qregister,cregister)
re_circuit.ry(alpha,1) # Prepare |psi_alpha> state
re_circuit.h(0)
re_circuit.append(SGate().control(),[0,1])
re_circuit.h(0)
re_circuit.measure(0,0)
re_circuit.draw()

# Create Hadamard test circuit for imaginary part
im_circuit = QuantumCircuit(qregister,cregister)
im_circuit.ry(alpha,1) # Prepare |psi_alpha> state
im_circuit.h(0)
im_circuit.append(SGate().inverse(),[0])
im_circuit.append(SGate().control(),[0,1])
im_circuit.h(0)
im_circuit.measure(0,0)
im_circuit.draw()

# Sample circuits over various alpha values
sampler = StatevectorSampler()
# Define primitive unified blocks for real and imaginary circuits
pub1 = (re_circuit, params)
pub2 = (im_circuit, params)
# Run two jobs with different numbers of shots
job1 = sampler.run([pub1, pub2],shots=shots1)
job2 = sampler.run([pub1, pub2],shots=shots2)
# Extract p(0) from results for real and imaginary circuits
re1 = [(job1.result()[0].data.classical.get_counts(i)['0']/shots1 * 2 - 1) for i in range(sweeps)]
im1 = [(job1.result()[1].data.classical.get_counts(i)['0']/shots1 * 2 - 1) for i in range(sweeps)]
re2 = [(job2.result()[0].data.classical.get_counts(i)['0']/shots2 * 2 - 1) for i in range(sweeps)]
im2 = [(job2.result()[1].data.classical.get_counts(i)['0']/shots2 * 2 - 1) for i in range(sweeps)]

# Get the exact solution we expect to verify results
# Function returning Ry matrix for a theta value
def ry_matrix(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],[np.sin(theta/2), np.cos(theta/2)]])
# Create statevectors for various theta values
psi_theta = []
for theta in _alpha:
    psi_theta.append(ry_matrix(theta)@[[1],[0]])
# Compute expectation values using matrix-vector multiplication
s_matrix = [[1,0],[0,1j]]
evs = []
for psi in psi_theta:
    evs.append(psi.T.conj() @ s_matrix @ psi)

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
plt.xlabel(r'$\alpha$')
ax1.set_ylabel(r'$Re( \langle \psi_\alpha | S | \psi_\alpha \rangle )$')
ax2.set_ylabel(r'$Im( \langle \psi_\alpha | S | \psi_\alpha \rangle )$')
ax1.scatter(_alpha,re1,color='r',marker='o')
ax2.scatter(_alpha,im1,color='r',marker='o')
ax1.scatter(_alpha,re2,color='b',marker='x')
ax2.scatter(_alpha,im2,color='b',marker='x')
ax1.plot(_alpha,np.array(np.real(evs)).flatten(),color='k')
ax2.plot(_alpha,np.array(np.imag(evs)).flatten(),color='k')
ax1.legend(['1000 shots','100,000 shots','Exact'],loc=9)
plt.show()

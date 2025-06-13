import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import ClassicalRegister
from qiskit.quantum_info.operators import Operator
from qiskit_aer import Aer
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.io import loadmat
from copy import deepcopy
from matplotlib import pyplot as plt

# Define the linear system:
# As an example, solve a matrix with numbers +,- 1...0.05  on the anti-diagonal,
# which is a non-Hermitian matrix
step = 0.05
A = np.diag(np.concatenate( (np.arange(-1,0,step),np.arange(step,1+step,step)) ) )
# Take anti-transpose of A to demonstrate the non-Hermitian matrix case.
A = np.flipud(A)
# Save matrix to check againt classical solution later
A_orig = deepcopy(A)    
# Normalize A. If an upper bound is known, use that instead.
# A = A/np.linalg.norm(A)

b = np.ones(np.shape(A)[1])
# Turn b into a quantum state
b = b/np.linalg.norm(b,2)
b_orig = deepcopy(b)

# Hermitian Dilation: only if A is not Hermitian
if np.any(A != A.conj().T):
    A = np.block([
        [np.zeros(np.shape(A)),A],
        [A.conj().T,np.zeros(np.shape(A))]
    ])
    b = np.block([
        b,
        np.zeros(np.shape(b))
    ])
    HD = True
else:
    HD = False    

# The matrix A needs to padded to some 2^n to enable block-encoding
if np.size(A)>1:
    A_num_qubits = int(np.ceil(np.log2(np.shape(A)[0])))
    padding_size = 2**A_num_qubits - np.shape(A)[0]
    if padding_size > 0:
        A = np.block([
            [A, np.zeros([np.shape(A)[0],padding_size])],
            [np.zeros([padding_size,np.shape(A)[0]]), np.zeros([padding_size,padding_size])]
        ])
else:
    A_num_qubits = 1
    padding_size = 1
    A = np.array([[A,0],[0,0]])
# Similarly, pad b
b = np.pad(b,(0,padding_size))

# Define the block-encoding of the matrix A
# If you have an efficient circuit to realize U_A (or O_A), use it here
U_A = np.block([
    [A   ,   -fractional_matrix_power(np.eye(np.shape(A)[0]) - np.linalg.matrix_power(A,2),0.5)],
    [fractional_matrix_power(np.eye(np.shape(A)[0]) - np.linalg.matrix_power(A,2),0.5)   ,   A]
])
# We also need to get the block-encoding size, i.e. m, used to encode A in U_A
m = int(np.log2(np.shape(U_A)[0]) - A_num_qubits)
U_A_num_qubits = int(np.log2(np.shape(U_A)[0]))
# Create the operator U_A in Qiskit
operatorA = Operator(U_A)

# Create the three registers for QSP:
# 1) 1 Z rotation qubit
# 2) m block-encoding ancillae
# 3) register for b
register_1 = QuantumRegister(size = 1, name = '|0>')
register_2 = QuantumRegister(size = m, name = '|0^m>')
register_3 = QuantumRegister(size = U_A_num_qubits-m, name = '|\phi>')

# Create a rotation circuit in the block-encoding basis
def CR_phi_d(phi, d, register_1, register_2):
    circuit = QuantumCircuit(register_1,register_2,name = 'CR_( \phi \tilde {})'.format(d))
    circuit.cx(register_2,register_1,ctrl_state=0)
    circuit.rz(phi*2, register_1)
    # Done this way for numerical stability
    circuit.z(register_1)
    circuit.cx(register_2,register_1,ctrl_state=0)
    return circuit

# Load QSP angles
# These angles can be obtained from the QSPPACK package
phi_angles = np.array( loadmat('phi_kappa_80_pts_8000_deg_1999.mat') ).item()['phi_proc']

phi_tilde_angles = np.zeros(np.shape(phi_angles))
phase_angles = phi_angles.reshape(phi_angles.shape[0])

# Create QSP circuit
QSP_circuit = QuantumCircuit(register_1, register_2, register_3, name = 'QSP')
# Initialize state |b>. If you have an efficient implementation for b, it goes here
QSP_circuit.initialize(b,list(reversed(register_3)))

# First Hadamard the ancilla qubit since we want Re(P(A))
QSP_circuit.h(register_1)
# Note: QSPPACK produces symmetric phase angles, so reversing phase angles is unnecessary
for d, phi in reversed(list(enumerate(phase_angles))):
    QSP_circuit = QSP_circuit.compose(CR_phi_d(phi,d,register_1,register_2))
    if d>(0):
        # The endianness of the bits matters. Need to change the order of the bits
        if d%2:
            QSP_circuit.append(operatorA.adjoint(),list(reversed(register_3[:])) + register_2[:])
        else:
            QSP_circuit.append(operatorA,list(reversed(register_3[:])) + register_2[:])

# Apply the final Hadamard gate
QSP_circuit.h(register_1)
# Account for little vs. big endian
QSP_circuit = QSP_circuit.reverse_bits()

# Run statevector simulator
solver='statevector'
backend = Aer.get_backend('statevector_simulator',precision = "double")
job = backend.run(QSP_circuit, shots=0)

# Extract relevant portion of statevector
QSP_statevector = job.result().get_statevector()
if HD:
    P_A_b = np.real(QSP_statevector.data[int(b_orig.shape[0]):(2*b_orig.shape[0])])
else:
    P_A_b = np.real(QSP_statevector.data[0:b.shape[0]])
P_A_b = P_A_b/np.linalg.norm(P_A_b)

# Get expected result using classical solver
expected_P_A_b = np.linalg.solve(A_orig,b_orig)
expected_P_A_b = expected_P_A_b/np.linalg.norm(expected_P_A_b)

# Plot QSP polynomial
x = np.flipud(A_orig).diagonal()
fig, ax1 = plt.subplots()
ax1.set_title('QSP QLSA')
ax1.scatter(x,P_A_b/P_A_b[-1],marker='x',c='g')
ax1.scatter(x,expected_P_A_b/expected_P_A_b[-1],marker='o',facecolors='none', edgecolors='k')
ax1.set_ylabel('P(x), 1/x')
plt.legend(['P(x)','1/x'],loc = 2)
ax2 = ax1.twinx()
ax2.plot(x[:x.size//2],np.log10(np.abs((P_A_b[:x.size//2]-expected_P_A_b[:x.size//2])/expected_P_A_b[-1])),'r')
ax2.plot(x[x.size//2:],np.log10(np.abs((P_A_b[x.size//2:]-expected_P_A_b[x.size//2:])/expected_P_A_b[-1])),'r')
ax2.set_ylim(bottom=-12, top=0)
ax2.set_ylabel('log10 |P(x)-1/x|')
plt.legend(['error'],loc = 1)
plt.xlabel('x')
plt.show()

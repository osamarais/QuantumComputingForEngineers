import qiskit
from qiskit import *
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat

print("Get phase angles from QSPPACK and store them as phi.mat")
print("Run this code by passing problem parameters in the following format:")
print("python this_code.py problem_number num_qubits_each_dimension num_iterations")

import sys
# print ('argument list', sys.argv)
# problem = int(sys.argv[1])
# n = int(sys.argv[2])
# l = int(sys.argv[3])

problem = 1
n = 2
l = 2

if problem == 1:
    d = 1
    NBCs = [[False, False]]
elif problem == 2:
    d = 1
    NBCs = [[False, True]]
elif problem == 3:
    d = 2
    NBCs = [[False, True],[False, False]]
elif problem == 4:
    d = 2
    NBCs = [[False, True],[False, True]]
elif problem == 5:
    d = 2
    NBCs = [[False, True],[True, True]]

print("problem number: {}".format(problem))
print("boundary conditions: {}".format(NBCs))
print("problem size: {}".format(n))
print("number of iterations {}".format(l))

# Load and prep angles
phi_angles = np.array( loadmat('phi_kappa_80_pts_8000_deg_1999.mat') ).item()['phi_proc']
phase_angles = phi_angles.reshape(phi_angles.shape[0])


########### Functions for C ####################3
def C_i(i,register):
    n = register.size
    if i<1 or i>(n-1):
        print('WRONG VALUE FOR i !!!!')
        return
    Ci = QuantumCircuit(register, name='C_{}'.format(i))
    [Ci.cx(control_qubit=i, target_qubit=(i-j), ctrl_state='0') for j in range(1,i+1)]
    Ci.append(qiskit.circuit.library.MCXGate(num_ctrl_qubits=i, ctrl_state='0'*i),register[:i+1])
#     Ci.mcx(control_qubits=workRegister[:i-1],target_qubit=workRegister[i-1])
    [Ci.cx(control_qubit=i, target_qubit=(i-j), ctrl_state='0') for j in reversed(range(1,i+1))]
    return Ci

############### Functions for R ###############
def L1_d(register):
    n = register.size
    
    # Circuit that creates L1 unitary
    L1 = QuantumCircuit(n,name='L1')
    L1.x(0)

    return L1


def L2_d(register):
    n = register.size

    L2 = QuantumCircuit(register,name='L2')
    for j in range(1,n):
        L2 = L2.compose(C_i(j,register))

    return L2

def L3_d(register,NBC):
    n = register.size

    L3 = QuantumCircuit(register,name='L3')

    if not NBC[0]:
        L3.x(0)
        L3.h(0)
        L3.append(qiskit.circuit.library.MCXGate(num_ctrl_qubits=n-1, ctrl_state='0'*(n-1)),register[1:]+[register[0]])
        L3.h(0)
        L3.x(0)

    if not NBC[1]:
        L3.h(n-1)
        L3.append(qiskit.circuit.library.MCXGate(num_ctrl_qubits=n-1, ctrl_state='1'*(n-1)),register)
        L3.h(n-1)

    return L3


def L4_d(register,NBC):
    n = register.size
    
    L4 = QuantumCircuit(register,name='L4')

    # Apply -ve sign to the unitary
    L4.z(0)
    L4.x(0)
    L4.z(0)
    L4.x(0)

    return L4

def R_circuit(n,d,workRegisters,lcuRegister,NBCs,alphas):

    # Create the Prep Circuit
    prep = QuantumCircuit(lcuRegister,name='Prep')
    prep.prepare_state(alphas)
    
    allregisters = []
    allregisters.extend(workRegisters)
    allregisters.extend([lcuRegister])

    blockEncoded = QuantumCircuit(*workRegisters,lcuRegister,name='R')

    # Apply the PREP operation
    blockEncoded = blockEncoded.compose(prep,lcuRegister)

    # Apply the SELECT operation using controlled versions of the circuits L1-L3
    # This needs to be done for each dimension!
    for i in range(d):
        if d>1:
            d_string = format(i, '0{}b'.format(d_size))
        else:
            d_string = ''
        blockEncoded.append(L1_d(workRegisters[i]).control(num_ctrl_qubits=(d_size+2)    ,ctrl_state=d_string+'00'), lcuRegister[:] + workRegisters[i][:])
        blockEncoded.append(L2_d(workRegisters[i]).control(num_ctrl_qubits=(d_size+2)    ,ctrl_state=d_string+'01'), lcuRegister[:] + workRegisters[i][:])
        blockEncoded.append(L3_d(workRegisters[i],NBCs[i]).control(num_ctrl_qubits=(d_size+2),ctrl_state=d_string+'10'), lcuRegister[:] + workRegisters[i][:])
        blockEncoded.append(L4_d(workRegisters[i],NBCs[i]).control(num_ctrl_qubits=(d_size+2),ctrl_state=d_string+'11'), lcuRegister[:] + workRegisters[i][:])
        
    # Apply the PREP+ operation
    blockEncoded = blockEncoded.compose(prep.inverse(),lcuRegister)
    
    return blockEncoded
    

######################## Functions for D ##########################
def L1_1(register):
    n = register.size
    
    # Circuit that creates L1 unitary
    L1 = QuantumCircuit(n,name='L1_1')
    L1.x(0)
    L1.z(0)
    
    return L1

def L1_2(register):
    n = register.size
    
    # Circuit that creates L1 unitary
    L1 = QuantumCircuit(n,name='L1_2')
    L1.z(0)
    L1.x(0)
    L1.z(0)

    return L1

def C_i(i,register):
    n = register.size
    if i<1 or i>(n-1):
        print('WRONG VALUE FOR i !!!!')
        return
    Ci = QuantumCircuit(register, name='C_{}'.format(i))
    [Ci.cx(control_qubit=i, target_qubit=(i-j), ctrl_state='0') for j in range(1,i+1)]
    Ci.append(qiskit.circuit.library.MCXGate(num_ctrl_qubits=i, ctrl_state='0'*i),register[:i+1])
    [Ci.cx(control_qubit=i, target_qubit=(i-j), ctrl_state='0') for j in reversed(range(1,i+1))]
    return Ci

def L2_1(register):
    n = register.size

    L2 = QuantumCircuit(register,name='L2_1')
    # Apply -ve to sign to alternating bits
    L2.z(0)
    for j in range(1,n):
        L2 = L2.compose(C_i(j,register))

    return L2

def L2_2(register):
    n = register.size

    L2 = QuantumCircuit(register,name='L2_2')
    L2.h(n-1)
    L2.append(qiskit.circuit.library.MCXGate(num_ctrl_qubits=n-1, ctrl_state='1'*(n-1)),register)
    L2.h(n-1)
    for j in range(1,n):
        L2 = L2.compose(C_i(j,register))
    L2.z(0)
    L2.x(0)
    L2.z(0)
    L2.x(0)
    
    return L2

def D_circuit(l,indexRegister,lcuRegister,alphas):

    # Create the Prep Circuit
    prep = QuantumCircuit(lcuRegister,name='Prep')
    prep.prepare_state(alphas)
    
    allregisters = []
    allregisters.extend([indexRegister])
    allregisters.extend([lcuRegister])

    blockEncoded = QuantumCircuit(indexRegister,lcuRegister,name='D')

    # Apply the PREP operation
    blockEncoded = blockEncoded.compose(prep,lcuRegister)

    # Apply the SELECT operation using controlled versions of the circuits L1_i, L2_i
    blockEncoded.append(L1_1(indexRegister).control(num_ctrl_qubits=(2)    ,ctrl_state='00'), lcuRegister[:] + indexRegister[:])
    blockEncoded.append(L1_2(indexRegister).control(num_ctrl_qubits=(2)    ,ctrl_state='01'), lcuRegister[:] + indexRegister[:])
    blockEncoded.append(L2_1(indexRegister).control(num_ctrl_qubits=(2)    ,ctrl_state='10'), lcuRegister[:] + indexRegister[:])
    blockEncoded.append(L2_2(indexRegister).control(num_ctrl_qubits=(2)    ,ctrl_state='11'), lcuRegister[:] + indexRegister[:])

    # Apply the PREP+ operation
    blockEncoded = blockEncoded.compose(prep.inverse(),lcuRegister)
    
    return blockEncoded

################ Functions for QSP Rotation ##################################
def CR_phi_d_efficient(phi, _d, signalReg, lcuRegister_R, lcuRegister_l, lcuRegister_q, circuit):
    BE_size = lcuRegister_R.size + lcuRegister_l.size + lcuRegister_q.size
    ctrl_state = '0'*(BE_size)
    
    circuit.append(qiskit.circuit.library.MCXGate(num_ctrl_qubits=BE_size, ctrl_state=ctrl_state), lcuRegister_R[:] + lcuRegister_l[:] + lcuRegister_q[:] + signalReg[:])

    circuit.rz(2*phi, signalReg)
    circuit.z(signalReg)
    
    circuit.append(qiskit.circuit.library.MCXGate(num_ctrl_qubits=BE_size, ctrl_state=ctrl_state), lcuRegister_R[:] + lcuRegister_l[:] + lcuRegister_q[:] + signalReg[:])

    return


####################### Construct qRLS Circuit ##############################
d_size = int(np.log2(d))

workRegisters = [QuantumRegister(n,name='dim {}'.format(i)) for i in range(d)]
indexRegister = QuantumRegister(l,name='index')

alphas_R = np.array([np.sqrt(1+0j), np.sqrt(1+0j), np.sqrt(0.5+0j), np.sqrt(0.5+0j)]*d)
alphas_R = alphas_R/np.linalg.norm(alphas_R,2)
lcu_size_R = int(np.ceil(np.log2(alphas_R.size)))
lcuRegister_R = QuantumRegister(lcu_size_R,name='lcu_R')
R = R_circuit(n,d,workRegisters,lcuRegister_R,NBCs,alphas_R)

alphas_l = np.array([np.sqrt(0.5+0j), np.sqrt(0.5+0j), np.sqrt(0.5+0j), np.sqrt(0.5+0j)])
alphas_l = alphas_l/np.linalg.norm(alphas_l,2)
lcu_size_l = int(np.ceil(np.log2(alphas_l.size)))
lcuRegister_l = QuantumRegister(lcu_size_l,name='lcu_l')
D = D_circuit(l,indexRegister,lcuRegister_l,alphas_l)

alphas_q = np.array([np.sqrt(1+0j), np.sqrt(3+0j)])
alphas_q = alphas_q/np.linalg.norm(alphas_q,2)
lcu_size_q = int(np.ceil(np.log2(alphas_q.size)))
lcuRegister_q = QuantumRegister(lcu_size_q,name='lcu_q')

R = R_circuit(n,d,workRegisters,lcuRegister_R,NBCs,alphas_R)
D = D_circuit(l,indexRegister,lcuRegister_l,alphas_l)
R = R.decompose(reps=6)
D = D.decompose(reps=6)

prep = QuantumCircuit(lcuRegister_q,name='Prep')
prep.prepare_state(alphas_q)
qRLS_circuit = QuantumCircuit(*workRegisters,indexRegister,lcuRegister_R,lcuRegister_l,lcuRegister_q,name='qRLS')
qRLS_circuit = qRLS_circuit.compose(prep,lcuRegister_q)
qRLS_circuit.append(R.control(num_ctrl_qubits=(1)    ,ctrl_state='1'), lcuRegister_q[:] + [_x for _xs in workRegisters for _x in _xs] + lcuRegister_R[:])
qRLS_circuit.append(D.control(num_ctrl_qubits=(1)    ,ctrl_state='1'), lcuRegister_q[:] + indexRegister[:]  + lcuRegister_l[:] )
qRLS_circuit = qRLS_circuit.compose(prep.inverse(),lcuRegister_q)

U_A = qRLS_circuit
U_A_i = U_A.inverse()

signalRegister = QuantumRegister(1,name='QSP signal')

QSP_circuit = QuantumCircuit(*workRegisters,indexRegister,lcuRegister_R,lcuRegister_l,lcuRegister_q,signalRegister,name='QSP_Solver')

# Prepare initial state
initial_state = np.ones(2**l)
initial_state[0] = 0
initial_state = initial_state/np.linalg.norm(initial_state,2)

QSP_circuit.append(qiskit.circuit.library.StatePreparation(initial_state),indexRegister)

for _i in workRegisters:
    QSP_circuit.h(_i)

####################### Start QSP Sequence #######################
# First thing is to  Hadamard the signal qubit since we want Re(P(A))
QSP_circuit.h(signalRegister)

for _d, phi in reversed( list(  enumerate(  phase_angles[:]))):
    CR_phi_d_efficient(phi,_d,signalRegister,lcuRegister_R,lcuRegister_l,lcuRegister_q,QSP_circuit)
    if _d>(0):
        if _d%2:
            for _ci in U_A_i.data:
                QSP_circuit.append(_ci)
        else:
            for _ci in U_A.data:
                QSP_circuit.append(_ci)
        
# Apply the final Hadamard gate
QSP_circuit.h(signalRegister)

####################### Simulate Circuit #######################
print('running simulation')
from qiskit_aer import Aer
device = 'CPU'
backend = Aer.get_backend('statevector_simulator', device=device, precision='double')

print('transpiling circuit')
transpiled_QSP_circuit = qiskit.transpile(QSP_circuit.decompose(reps=3))

print('completed transpilation, starting job')
result = backend.run(transpiled_QSP_circuit,shots=0).result()

statevector = result.get_statevector()
final_output = np.array(statevector)[0:(2**(l)*2**(n*d))]
iterates = final_output/np.linalg.norm(final_output,2)

####################### Calculate classical iterates #######################
R = [np.zeros((2**n,2**n)) for _i in NBCs]
for _R in R:
    i, j = np.indices(_R.shape)
    _R[i==j-1] = 0.5
    _R[i==j+1] = 0.5
for _i,_R in zip(NBCs,R):
    if _i[0]:
        _R[0,0] = 0.5
    if _i[1]:
        _R[-1,-1] = 0.5
R_full = np.zeros((2**(n*d),2**(n*d)))
for _i,_R in enumerate(reversed(R)):
    if _i == 0:
        _Ri = _R
    else:
        _Ri = np.eye(2**n)
    for _j in range(1,d):
        if _j==_i:
            _Ri = np.kron(_Ri,_R)
        else:
            _Ri = np.kron(_Ri,np.eye(2**n))
    R_full += _Ri
R_full = R_full/d

classical_iterates = [np.zeros(R_full.shape[0]) for _i in range(2**l)]
f = np.ones(R_full.shape[0])
for _i in range(2**l - 1):
    classical_iterates[_i+1] = np.matmul(R_full,classical_iterates[_i]) + f
classical_iterates = np.concatenate(classical_iterates)
classical_iterates = classical_iterates/np.linalg.norm(classical_iterates,2)

####################### Calculate the exact solution #######################
A = [np.zeros((2**n,2**n)) for _i in NBCs]
for _A in A:
    i, j = np.indices(_A.shape)
    _A[i==j-1] = -1
    _A[i==j+1] = -1
    _A[i==j] = 2
for _i,_A in zip(NBCs,A):
    if _i[0]:
        _A[0,0] = 1
    if _i[1]:
        _A[-1,-1] = 1
A_full = np.zeros((2**(n*d),2**(n*d)))
for _i,_A in enumerate(reversed(A)):
    if _i == 0:
        _Ai = _A
    else:
        _Ai = np.eye(2**n)
    for _j in range(1,d):
        if _j==_i:
            _Ai = np.kron(_Ai,_A)
        else:
            _Ai = np.kron(_Ai,np.eye(2**n))
    A_full += _Ai
f = np.ones(R_full.shape[0])
exact_sol = np.linalg.solve(A_full,f)
exact_sol = exact_sol/np.linalg.norm(exact_sol,2)

####################### Calculate the iterate errors #######################
quantum_convergence = []
for _i in range(1,2**l):
    quantum_convergence.append( np.linalg.norm(exact_sol - iterates[(2**(n*d))*(_i):(2**(n*d))*(_i+1)]/np.linalg.norm(iterates[(2**(n*d))*(_i):(2**(n*d))*(_i+1)],2)) )

classical_convergence = []
for _i in range(1,2**l):
    classical_convergence.append( np.linalg.norm(exact_sol - classical_iterates[(2**(n*d))*(_i):(2**(n*d))*(_i+1)]/np.linalg.norm(classical_iterates[(2**(n*d))*(_i):(2**(n*d))*(_i+1)],2)) )

####################### Calculate the QSP errors #######################
QSP_errors_full = []
for _i in range(2**l):
    QSP_errors_full.append( np.linalg.norm( ( classical_iterates[(2**(n*d))*(_i):(2**(n*d))*(_i+1)]) - iterates[_i*(2**(n*d)):(_i+1)*(2**(n*d))], 2 ) )

####################### Save all useful variables #######################
mdic = {"dimensions":d, "n":n, "l":l, "BCs":NBCs, "raw_statevector": final_output, "classical_solution":classical_iterates, "QSP_solution":iterates, "QSP_Errors":QSP_errors_full, "classical":classical_convergence, "quantum":quantum_convergence}
savemat("output_problem_{}_n_{}_l_{}.mat".format(problem,n,l), mdic)

print("completed simulation with parameters:")
print("problem number: {}".format(problem))
print("boundary conditions: {}".format(NBCs))
print("problem size: {}".format(n))
print("number of iterations {}".format(l))

print('\n\nsaved output\n\n')


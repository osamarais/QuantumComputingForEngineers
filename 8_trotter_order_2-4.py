#!/usr/bin/python3

import qiskit
from qiskit_aer import UnitarySimulator
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter

min_t = 1
max_t = 5
# Number of Trotter Steps
m = [1,2,4,8,16,32,64,128]

myOp = SparsePauliOp("XYZ",0.9) + SparsePauliOp("YZX",1.1)

# 2nd order Suzuki-Trotter
for t in range(min_t,max_t+1):
    # Calculate exact solution classically
    exact_solution = expm( -t * 1j * myOp.to_matrix() )

    errors = []
    for r in m:
        # Define Suzuki-Trotter method
        mySynthesis = SuzukiTrotter(order=2, reps=r,  cx_structure='chain')
        # Create Pauli Evolution
        myEvolution = PauliEvolutionGate(myOp,time=t,synthesis=mySynthesis)
        # Append to a quantum circuit
        myCircuit = qiskit.QuantumCircuit(myOp.num_qubits)
        myCircuit.append(myEvolution,range(0,myOp.num_qubits))

        # Simulate the circuit to obtain overall unitary of Trotterization
        mySimulator = UnitarySimulator()
        result = mySimulator.run(myCircuit.decompose(reps=2)).result()
        finalUnitary = result.get_unitary()

        # Compare circuit unitary with exact unitary
        errors.append( np.linalg.norm(finalUnitary - exact_solution,2) )

    plt.loglog(m,errors)
    
# 4th order Suzuki-Trotter
for t in range(min_t,max_t+1):
    # Calculate exact solution classically
    exact_solution = expm( -t * 1j * myOp.to_matrix() )

    errors = []
    for r in m:
        # Define Suzuki-Trotter method
        mySynthesis = SuzukiTrotter(order=4, reps=r,  cx_structure='chain')
        # Create Pauli Evolution
        myEvolution = PauliEvolutionGate(myOp,time=t,synthesis=mySynthesis)
        # Append to a quantum circuit
        myCircuit = qiskit.QuantumCircuit(myOp.num_qubits)
        myCircuit.append(myEvolution,range(0,myOp.num_qubits))

        # Simulate the circuit to obtain overall unitary of Trotterization
        mySimulator = UnitarySimulator()
        result = mySimulator.run(myCircuit.decompose(reps=2)).result()
        finalUnitary = result.get_unitary()

        # Compare circuit unitary with exact unitary
        errors.append( np.linalg.norm(finalUnitary - exact_solution,2) )

    plt.loglog(m,errors,linestyle='dashed')
plt.xlabel('# of Trotter steps')
plt.ylabel(r'$|error|_2$')
plt.legend(['t = {}, 2nd Order'.format(t) for t in range(min_t,max_t+1)]
	+['t = {}, 4th Order'.format(t) for t in range(min_t,max_t+1)])
plt.show()